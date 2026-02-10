# NJ Golf Course Mapper & Validator

A Python pipeline that discovers, cross-validates, and maps all golf courses in New Jersey using OpenStreetMap polygons, Foursquare Open Source Places POIs, and web sources.

## Overview

This project combines three independent data sources to build a comprehensive picture of golf courses in NJ:

1. **OpenStreetMap (OSM)** -- polygon boundaries tagged `leisure=golf_course`
2. **Foursquare OS Places** -- point-of-interest data from the open S3 dataset (no API key required)
3. **Web sources** -- scraped course details (type, status, acreage)

The pipeline cross-validates these sources, identifies courses that appear in only one dataset, and uses two machine learning models to assess whether Foursquare-only POIs represent real golf courses.

## Results Summary

| Metric | Count |
|--------|-------|
| OSM golf course polygons | 314 |
| Foursquare golf course POIs | 756 |
| Matched (OSM + Foursquare) | 290 (92.4%) |
| OSM-only (no Foursquare match) | 24 |
| Foursquare-only (no OSM polygon) | 483 |
| V1 likely golf courses (land cover) | 425/483 |
| V2 likely golf courses (building analysis) | 353/483 |
| Web-validated courses | 221 |

## Output Files

All outputs are in the `data/` directory:

| File | Description | Size |
|------|-------------|------|
| `nj_golf_courses.geojson` | OSM polygon geometries | ~2.0 MB |
| `foursquare_golf_courses.parquet` | Foursquare POI data | ~65 KB |
| `foursquare_validation_report.csv` | Cross-validation with V1 scores | ~101 KB |
| `foursquare_validation_report_v2.csv` | Cross-validation with V1 + V2 scores | ~112 KB |
| `nj_golf_courses_map.html` | Interactive map (standalone) | ~1.8 MB |
| `nj_golf_courses_table.csv` | Summary table of all OSM courses | ~47 KB |
| `web_validation_report.md` | Web validation report | ~27 KB |

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: osmnx, geopandas, folium, duckdb, requests, beautifulsoup4, shapely, pyproj, pandas, branca, scikit-learn, numpy.

## Usage

```bash
python nj_golf_courses.py
```

The pipeline runs all steps sequentially and prints progress to stdout. Optionally set `MAPBOX_ACCESS_TOKEN` as an environment variable to enable Mapbox tile layers on the map.

## Pipeline Steps

### Step 1: Fetch OSM Golf Course Polygons

Queries the Overpass API for all `leisure=golf_course` ways and relations within New Jersey. Uses POST requests with a 300-second timeout and falls back to `overpass.kumi.systems` if the primary server fails. Parses the JSON response into a GeoDataFrame with proper polygon geometries (including multipolygon relations).

### Step 2: Fetch Foursquare OS Places Data

Uses DuckDB with the `httpfs` extension to query Foursquare's public S3 parquet files directly (no download of the full ~11GB dataset). Filters for:
- `country = 'US'` and `region = 'NJ'`
- `fsq_category_labels` containing "golf course"
- Excludes "mini golf" POIs

S3 path: `s3://fsq-os-places-us-east-1/release/dt=2025-09-09/places/parquet/**/*.parquet`

### Step 3: Cross-Validation (OSM vs Foursquare)

For each OSM polygon, searches for Foursquare POIs that either:
- Fall **inside** the polygon (match method: `contained`)
- Are **within 500m** of the polygon boundary (match method: `proximity`)

Uses `EPSG:32618` (UTM Zone 18N) for accurate distance calculations. Invalid geometries are repaired with `shapely.validation.make_valid()`.

Result categories:
- **Matched**: OSM polygon has a corresponding Foursquare POI
- **OSM-only**: OSM polygon with no Foursquare match
- **Foursquare-only**: Foursquare POI with no matching OSM polygon

### Step 3b: V1 -- Land Cover Analysis

Scores Foursquare-only POIs based on surrounding green land cover from OSM.

**Approach:**
1. Fetch all green/natural land cover polygons in NJ from Overpass (forest, farmland, grass, meadow, recreation ground, park, garden, nature reserve, wood, scrub, grassland, heath) -- 75,154 polygons
2. For each POI, compute features within a 500m buffer:
   - `green_ratio`: Fraction of buffer area covered by green land
   - `max_patch_ratio`: Largest single green patch as fraction of buffer
   - `patch_count`: Number of distinct green patches
3. Train a `GradientBoostingClassifier` using:
   - **Positive examples**: 290 centroids of verified OSM+Foursquare matched courses
   - **Negative examples**: 200 random points in NJ, each >2km from any known course
4. Apply the model to all 483 Foursquare-only POIs

**Results:**
- Training accuracy: 93.1%
- Feature importances: `green_ratio=0.73`, `patch_count=0.18`, `max_patch_ratio=0.09`
- 425/483 scored as likely (>= 50%)

**Limitation:** V1 depends on OSM land cover mapping completeness. Areas with sparse land-use mapping score lower regardless of actual conditions (e.g., Union League National at 26%, Sand Barrens at 26%).

### Step 3c: V2 -- Building-Based Directional Ellipse Analysis

Scores Foursquare-only POIs based on building absence in candidate golf course areas.

**Motivation:** V1's reliance on mapped green areas fails where OSM coverage is sparse. V2 uses a complementary signal: real golf courses have very few buildings, while urban POIs (indoor golf, simulators) are surrounded by dense development.

**Approach:**

**Phase 1 -- Learn golf course geometry from verified matches:**
- Analyze 290 verified OSM+Foursquare courses to learn typical geometry:
  - Median area: 501,203 sq m (124 acres)
  - Median aspect ratio: 1.34
  - POI position: 77% of POIs fall inside their polygon; median offset from centroid is 322m
- Conclusion: The Foursquare POI (clubhouse) typically sits at the **edge** of the golf course, not the center

**Phase 2 -- Fetch building footprints:**
- Query Overpass for building center points around all POI locations
- Batch queries (10 points per batch) with 1.5s rate limiting
- Search radius: 1,185m per point
- Result: 213,177 unique building points across NJ

**Phase 3 -- Directional ellipse scoring:**
- For each POI, place a golf-course-sized ellipse (462m x 346m semi-axes) in 16 compass directions with the POI at the ellipse edge
- Count buildings inside each candidate ellipse
- Record the **minimum** building count across all 16 directions (best-case orientation)
- Compute building density = min_buildings / ellipse_area

**Phase 4 -- Train classifier:**
- Features: `min_buildings_in_ellipse`, `density`
- Same positive/negative training split as V1 (subset of 80 verified, 50 negative)
- Training accuracy: 93.1%
- Feature importances: `density=0.86`, `min_buildings=0.14`
- 353/483 scored as likely (>= 50%)

### V1 vs V2 Comparison -- Key POIs

| Course Name | V1 | V2 | Notes |
|-------------|----|----|-------|
| Avalon Golf Club | 85% | 60% | Real course; both models identify it |
| Union League National Golf Club | 26% | **98%** | Real course; V1 failed due to sparse OSM land cover, V2 correctly detects open area |
| Sand Barrens Golf Club | 26% | **98%** | Real course; same V1 limitation fixed by V2 |
| Cape May Par 3 Golf | 82% | 29% | Small par-3; V2 detects nearby buildings (small course surrounded by development) |
| Pinelands Golf Course | 99% | 65% | Real course; both models agree, V2 slightly more conservative |
| Indian Spring Golf Course | 27% | **98%** | Real course; V2 improvement |
| Post Brook Farms Country Club | 3% | **92%** | Real course; dramatic V2 improvement |
| Hoboken Golf | 99% | **2%** | Indoor/simulator; V2 correctly identifies urban location |
| Golf Simulator | 99% | **4%** | Indoor; V2 correctly identifies urban location |
| Atlantic City Miniature Golf | 85% | **9%** | Miniature golf; V2 correctly identifies dense area |

**Key takeaway:** V1 and V2 are complementary. V1 excels where OSM land cover is well-mapped; V2 excels at identifying courses in areas with sparse land cover mapping and at rejecting urban/indoor golf venues that V1 incorrectly scores high.

### Step 4: Web Validation

For each OSM golf course, searches the web for:
- Official reported acreage/area
- Whether the course is currently open or closed
- Course type (public/private, 9-hole/18-hole)

Uses `requests` + `BeautifulSoup` to scrape search results. Results saved to `data/web_validation_report.md`.

### Step 5: Interactive Map

Creates `data/nj_golf_courses_map.html` using Folium with:

**Base layers (toggle via layer control):**
- OpenStreetMap
- Google Roads
- Google Satellite
- Google Hybrid
- Mapbox (if `MAPBOX_ACCESS_TOKEN` is set)

**Overlay layers (toggleable):**
- **Verified (green polygons):** OSM courses confirmed by Foursquare
- **OSM Only (orange polygons):** OSM courses with no Foursquare match
- **Foursquare Only (blue markers):** Foursquare POIs with no OSM polygon (filtered to >= 50% probability)

**Popups** show course name, area (sq m and acres), Foursquare match status, and web-validated info when available.

### Step 6-7: Summary Tables

Generates `data/nj_golf_courses_table.csv` with columns: name, area_sq_meters, area_acres, latitude, longitude, osm_id, foursquare_verified, foursquare_name, foursquare_fsq_place_id, web_validated, course_type, status.

## Architecture

### Module Structure (`nj_golf_courses.py`)

| Function | Description |
|----------|-------------|
| `fetch_osm_golf_courses()` | Overpass API query, returns GeoDataFrame |
| `fetch_foursquare_golf_courses()` | DuckDB S3 query, returns DataFrame |
| `validate_with_foursquare(osm_gdf, fsq_df)` | Spatial cross-validation |
| `fetch_nj_landcover()` | Fetch green/natural OSM land cover |
| `validate_foursquare_only(osm_gdf, fsq_validation)` | V1: land cover model |
| `_fetch_buildings_around_points(query_points, radius_m)` | Batch Overpass building queries |
| `validate_foursquare_only_v2(osm_gdf, fsq_validation, fsq_df)` | V2: building ellipse model |
| `validate_with_web(osm_gdf)` | Web scraping validation |
| `create_map(gdf, validation_df, web_validation)` | Folium map generation |
| `create_tables(gdf, fsq_validation, web_validation)` | CSV/report generation |
| `main()` | Orchestrates the full pipeline |

### Technical Notes

- **CRS**: All area calculations and spatial operations use EPSG:32618 (UTM Zone 18N), appropriate for New Jersey
- **Geometry repair**: `shapely.validation.make_valid()` fixes invalid OSM polygon geometries that cause TopologyException errors during spatial joins
- **Overpass rate limiting**: Building queries use 1.5s delay between batches of 10 points; falls back to secondary server on failure
- **DuckDB S3**: Uses `httpfs` extension with unsigned access (`s3_url_style='path'`, `s3_region='us-east-1'`) to query parquet files directly from S3 without downloading the full dataset
- **Mini golf exclusion**: Foursquare category filter uses `NOT LIKE '%mini golf%'` to exclude mini golf venues (reduced results from 852 to 756)

## Data Sources

- **OpenStreetMap**: [overpass-api.de](https://overpass-api.de) -- `leisure=golf_course` tags
- **Foursquare OS Places**: [S3 bucket](https://opensource.foursquare.com/os-places/) -- September 2025 release, public access, no API key
- **Web**: Various golf course directories and search results
