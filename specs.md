## Project: NJ Golf Course Mapper & Validator

### Goal
Find all golf courses in New Jersey, visualize their polygons on an interactive map, cross-validate with Foursquare Open Source Places data and web sources, and produce summary tables and validation reports.

### Directory Structure
- nj_golf_courses.py (main module)
- requirements.txt
- data/
  - nj_golf_courses.geojson
  - nj_golf_courses_map.html
  - nj_golf_courses_table.csv
  - foursquare_golf_courses.parquet
  - foursquare_validation_report.csv
  - web_validation_report.md

### Steps

1. **Fetch golf course polygons from OpenStreetMap** using osmnx or the Overpass API directly. Query for `leisure=golf_course` within New Jersey's boundary. Save the raw geodata to `data/nj_golf_courses.geojson`.

2. **Fetch golf course POIs from Foursquare Open Source Places** via their public S3 bucket. Use DuckDB to query the Parquet files directly from S3 without downloading the full dataset (~11GB). No API key is required.

   - S3 bucket: `s3://fsq-os-places-us-east-1/release/dt=2025-09-09/places/parquet/`
   - Use `--no-sign-request` / unsigned access
   - Filter for `country = 'US'` and `region = 'NJ'` (or use state-level geo filtering)
   - Filter categories containing "Golf" using the `fsq_category_labels` column
   - Also download the categories file from `s3://fsq-os-places-us-east-1/release/dt=2025-09-09/categories/parquet/` if needed to resolve category IDs
   - Save filtered results to `data/foursquare_golf_courses.parquet`
   - Note: Foursquare OS Places contains points only (lat/lon), not polygons

3. **Cross-validate OSM polygons against Foursquare points:**
   - For each Foursquare point, check if it falls within or near (within 500m) an OSM polygon using spatial join.
   - For each OSM polygon, check if any Foursquare point matches.
   - Generate `data/foursquare_validation_report.csv` with columns: osm_name, osm_id, area_sq_meters, area_acres, foursquare_match (boolean), foursquare_name, foursquare_fsq_place_id, distance_meters, match_method (contained/proximity/none).
   - Include a summary section at the top or bottom: count of OSM-only courses, Foursquare-only courses (not in any OSM polygon), and matched courses.

4. **Web validation:** For each golf course found, use web search or scraping to try to find:
   - Official reported acreage/area
   - Whether the course is currently open or closed
   - Course type (public/private, 9-hole/18-hole)
   - Generate `data/web_validation_report.md` as a readable markdown report with:
     - A table comparing OSM-computed area vs web-reported area (with % difference)
     - Notes on courses where area differs significantly (>20%)
     - Sources/URLs for each data point found
     - Summary statistics (how many validated, how many discrepancies found)
   - Use requests + BeautifulSoup to scrape search results, or a golf course directory like golflink.com, nj.com golf listings, or similar public sources.

5. **Create an interactive HTML map** (`data/nj_golf_courses_map.html`) using folium. Requirements:
   - OpenStreetMap base layer
   - Each golf course polygon rendered with semi-transparent fill
   - Color-code polygons: green = verified by Foursquare, orange = OSM only
   - Foursquare-only courses (no matching OSM polygon) shown as blue point markers
   - Popup on click showing: course name, area (sq meters and acres), Foursquare match status, web-validated info if available
   - Legend explaining the color coding
   - Should work standalone in any browser (no server needed)

6. **Compute area** for each polygon in square meters and acres (use EPSG:32618 / UTM Zone 18N for NJ).

7. **Generate summary table** saved as `data/nj_golf_courses_table.csv` with columns: name, area_sq_meters, area_acres, latitude, longitude, osm_id, foursquare_verified, foursquare_name, foursquare_fsq_place_id, web_validated, course_type, status.

8. **Create the Python module** as `nj_golf_courses.py` with functions:
   - `fetch_osm_golf_courses() -> GeoDataFrame`
   - `fetch_foursquare_golf_courses() -> DataFrame` (uses DuckDB to query S3 Parquet)
   - `validate_with_foursquare(osm_gdf, fsq_df) -> DataFrame`
   - `validate_with_web(osm_gdf) -> DataFrame`
   - `create_map(gdf, validation_df) -> saves HTML`
   - `create_tables(gdf, fsq_validation, web_validation) -> saves CSVs and reports`
   - `main()` that runs the full pipeline

9. **Add a `requirements.txt`** with all dependencies (osmnx, geopandas, folium, duckdb, requests, beautifulsoup4, shapely, pyproj, pandas, etc.).

10. **Run the full pipeline** end-to-end and confirm all output files are generated in `data/`. Print a final summary to stdout: total courses found from OSM, total from Foursquare, match rate, any errors encountered.

### Technical Notes
- The Foursquare S3 bucket requires no authentication. Use DuckDB's S3 httpfs extension with unsigned requests.
- The September 2025 release (dt=2025-09-09) is the latest available on the public bucket. Future releases require registration at the Foursquare Places Portal.
- Foursquare data has points only. OSM has polygons. The validation is checking whether Foursquare confirms the existence of a golf course at/near the OSM polygon location.
- For the DuckDB S3 query, install the httpfs extension and configure unsigned access:
  ```python
  import duckdb
  con = duckdb.connect()
  con.execute("INSTALL httpfs; LOAD httpfs;")
  con.execute("SET s3_region='us-east-1';")
  con.execute("SET s3_url_style='path';")
  ```

### Constraints
- Create `data/` subfolder for all outputs
- Pure Python, no notebooks
- Handle failures gracefully (skip web validation if sites block scraping, log errors)
- All files in the current directory
- Print progress to stdout as each step completes