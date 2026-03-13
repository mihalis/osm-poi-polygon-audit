"""
Configurable POI Pipeline — OSM + Foursquare Mapper & Validator

Fetches point-of-interest polygons from OpenStreetMap, cross-validates with
Foursquare, and produces interactive maps and summary reports.

By default, the pipeline targets NJ golf courses, but behavior is fully
configurable via the PipelineConfig dataclass.
"""

import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field

import duckdb
import folium
import geopandas as gpd
import pandas as pd
import requests
from branca.element import Template, MacroElement
from shapely.validation import make_valid

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MAPBOX_TOKEN = os.environ.get("MAPBOX_ACCESS_TOKEN", "")


@dataclass
class PipelineConfig:
    osm_tag: str = "leisure"
    osm_value: str = "golf_course"
    keyword: str = "Golf Course"
    exclude_keywords: list = field(default_factory=lambda: ["Mini Golf", "Miniature Golf", "Minigolf"])
    state_name: str = "New Jersey"
    state_abbrev: str = "NJ"

    @property
    def file_prefix(self):
        kw = self.keyword.lower().replace(" ", "_") + "s"
        return f"{self.state_abbrev.lower()}_{kw}"

    @property
    def display_name(self):
        return f"{self.state_name} {self.keyword}s"

    @property
    def unknown_name(self):
        return f"Unknown {self.keyword}"

    @property
    def utm_crs(self):
        state_centroids = {
            "NJ": -74.4, "NY": -75.5, "PA": -77.2, "CT": -72.7,
            "CA": -119.4, "TX": -99.9, "FL": -81.5, "IL": -89.4,
            "OH": -82.8, "GA": -83.6, "NC": -79.8, "MI": -84.5,
            "VA": -78.5, "WA": -120.7, "AZ": -111.1, "MA": -71.5,
            "TN": -86.6, "IN": -86.1, "MO": -91.8, "MD": -76.6,
            "WI": -89.6, "CO": -105.8, "MN": -94.6, "SC": -81.0,
            "AL": -86.9, "LA": -91.9, "KY": -84.3, "OR": -120.6,
        }
        lon = state_centroids.get(self.state_abbrev, -74.4)
        zone = int((lon + 180) / 6) + 1
        return f"EPSG:326{zone:02d}"

    def file_path(self, suffix):
        return f"{self.file_prefix}_{suffix}"


def _make_hexagon(lat, lon, radius_m=75):
    """Generate a flat-top hexagon polygon (GeoJSON) centered on lat/lon."""
    lat_off = radius_m / 111000
    lon_off = radius_m / (111000 * math.cos(math.radians(lat)))
    coords = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        coords.append([lon + lon_off * math.cos(angle), lat + lat_off * math.sin(angle)])
    coords.append(coords[0])
    return {"type": "Polygon", "coordinates": [coords]}


def _count_polygon_vertices(geojson_geom):
    """Count vertices in a GeoJSON geometry dict."""
    t = geojson_geom["type"]
    if t == "Polygon":
        return sum(len(ring) for ring in geojson_geom["coordinates"])
    elif t == "MultiPolygon":
        return sum(len(ring) for poly in geojson_geom["coordinates"] for ring in poly)
    elif t == "GeometryCollection":
        return sum(_count_polygon_vertices(g) for g in geojson_geom["geometries"])
    return 0


def fetch_osm_golf_courses(config=None):
    """Fetch POI polygons from OpenStreetMap via Overpass API."""
    if config is None:
        config = PipelineConfig()
    print(f"[1/5] Fetching {config.keyword.lower()} polygons from OpenStreetMap...")

    overpass_servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    query = f"""
    [out:json][timeout:300];
    area["name"="{config.state_name}"]["admin_level"="4"]->.searchArea;
    (
      way["{config.osm_tag}"="{config.osm_value}"](area.searchArea);
      relation["{config.osm_tag}"="{config.osm_value}"](area.searchArea);
    );
    out body;
    >;
    out skel qt;
    """

    data = None
    for overpass_url in overpass_servers:
        try:
            print(f"  Trying {overpass_url}...")
            response = requests.post(
                overpass_url, data={"data": query}, timeout=360
            )
            response.raise_for_status()
            data = response.json()
            print(f"  Success from {overpass_url}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if data is None:
        raise RuntimeError("All Overpass API servers failed")

    nodes = {}
    ways = {}
    relations = {}

    for element in data["elements"]:
        if element["type"] == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])
        elif element["type"] == "way":
            ways[element["id"]] = element
        elif element["type"] == "relation":
            relations[element["id"]] = element

    features = []

    # Process ways
    for way_id, way in ways.items():
        # Skip ways that are part of a relation (they'll be handled there)
        coords = []
        nids = []  # parallel array of OSM node IDs
        for node_id in way.get("nodes", []):
            if node_id in nodes:
                coords.append(nodes[node_id])
                nids.append(node_id)
        if len(coords) < 3:
            continue

        # Ensure polygon is closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])
            nids.append(nids[0])

        name = way.get("tags", {}).get("name", config.unknown_name)
        tags = way.get("tags", {})

        feature = {
            "type": "Feature",
            "properties": {
                "name": name,
                "osm_id": f"way/{way_id}",
                "osm_type": "way",
                "_node_ids": [nids],  # one array per ring, parallel to coordinates
                **{k: v for k, v in tags.items() if k != "name"},
            },
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        }
        features.append(feature)

    # Process relations (multipolygon)
    for rel_id, rel in relations.items():
        name = rel.get("tags", {}).get("name", config.unknown_name)
        tags = rel.get("tags", {})

        outer_rings = []
        outer_nids = []  # parallel node ID arrays for each ring
        for member in rel.get("members", []):
            if member["type"] == "way" and member.get("role", "") == "outer":
                way = ways.get(member["ref"])
                if way:
                    coords = []
                    nids = []
                    for node_id in way.get("nodes", []):
                        if node_id in nodes:
                            coords.append(nodes[node_id])
                            nids.append(node_id)
                    if len(coords) >= 3:
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                            nids.append(nids[0])
                        outer_rings.append(coords)
                        outer_nids.append(nids)

        # Remove individual way features that are part of this relation
        member_way_ids = {
            m["ref"] for m in rel.get("members", []) if m["type"] == "way"
        }
        features = [
            f
            for f in features
            if not (
                f["properties"]["osm_type"] == "way"
                and int(f["properties"]["osm_id"].split("/")[1]) in member_way_ids
            )
        ]

        if not outer_rings:
            continue

        if len(outer_rings) == 1:
            geometry = {"type": "Polygon", "coordinates": outer_rings}
            node_ids_prop = outer_nids  # [[nid, nid, ...]]
        else:
            geometry = {
                "type": "MultiPolygon",
                "coordinates": [[ring] for ring in outer_rings],
            }
            node_ids_prop = [[nids] for nids in outer_nids]  # [[[nid, ...]]]

        feature = {
            "type": "Feature",
            "properties": {
                "name": name,
                "osm_id": f"relation/{rel_id}",
                "osm_type": "relation",
                "_node_ids": node_ids_prop,
                **{k: v for k, v in tags.items() if k != "name"},
            },
            "geometry": geometry,
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}

    geojson_path = os.path.join(DATA_DIR, config.file_path("data.geojson"))
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    print(f"  Found {len(gdf)} {config.keyword.lower()}s from OpenStreetMap.")
    print(f"  Saved to {geojson_path}")
    return gdf


def fetch_foursquare_golf_courses(config=None):
    """Fetch POIs from Foursquare Open Source Places via S3/DuckDB."""
    if config is None:
        config = PipelineConfig()
    print(f"[2/5] Fetching {config.keyword.lower()} data from Foursquare OS Places (S3)...")

    s3_path = "s3://fsq-os-places-us-east-1/release/dt=2025-09-09/places/parquet/**/*.parquet"

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-east-1';")
    con.execute("SET s3_url_style='path';")
    con.execute("SET s3_access_key_id='';")
    con.execute("SET s3_secret_access_key='';")

    print("  Querying Foursquare OS Places parquet files from S3...")
    print("  (This scans ~11GB of data remotely, may take a few minutes)")

    # Build exclude clauses dynamically
    exclude_clauses = []
    for ek in config.exclude_keywords:
        ek_lower = ek.lower()
        exclude_clauses.append(
            f"AND lower(array_to_string(fsq_category_labels, '|')) NOT LIKE '%{ek_lower}%'"
        )
        exclude_clauses.append(f"AND lower(name) NOT LIKE '%{ek_lower}%'")
    exclude_sql = "\n          ".join(exclude_clauses)

    df = con.execute(f"""
        SELECT fsq_place_id, name, latitude, longitude, address, locality,
               region, fsq_category_labels, date_closed
        FROM read_parquet('{s3_path}')
        WHERE country = 'US'
          AND region = '{config.state_abbrev}'
          AND fsq_category_labels IS NOT NULL
          AND lower(array_to_string(fsq_category_labels, '|')) LIKE '%{config.keyword.lower()}%'
          {exclude_sql}
    """).fetchdf()

    con.close()

    # Convert category labels list to a readable string
    df["fsq_category_labels"] = df["fsq_category_labels"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else str(x)
    )

    fsq_path = os.path.join(DATA_DIR, f"foursquare_{config.file_prefix}.parquet")
    df.to_parquet(fsq_path, index=False)
    print(f"  Found {len(df)} {config.keyword.lower()}s from Foursquare OS Places.")
    print(f"  Saved to {fsq_path}")
    return df


def validate_with_foursquare(osm_gdf, fsq_df, config=None):
    """Cross-validate OSM polygons against Foursquare points."""
    if config is None:
        config = PipelineConfig()
    print("[3/5] Cross-validating OSM data with Foursquare...")

    if fsq_df.empty:
        print("  No Foursquare data available. Generating OSM-only report.")
        osm_projected = osm_gdf.to_crs(config.utm_crs)
        records = []
        for idx, row in osm_gdf.iterrows():
            area = osm_projected.geometry.iloc[
                osm_gdf.index.get_loc(idx)
            ].area
            records.append(
                {
                    "osm_name": row.get("name", "Unknown"),
                    "osm_id": row.get("osm_id", ""),
                    "area_sq_meters": round(area, 2),
                    "area_acres": round(area * 0.000247105, 2),
                    "foursquare_match": False,
                    "foursquare_name": "",
                    "foursquare_fsq_place_id": "",
                    "distance_meters": None,
                    "match_method": "none",
                }
            )
        result = pd.DataFrame(records)
        report_path = os.path.join(DATA_DIR, config.file_path("validation_report.csv"))
        result.to_csv(report_path, index=False)
        print(f"  Saved validation report to {report_path}")
        print(f"  Summary: {len(records)} OSM-only, 0 Foursquare-only, 0 matched")
        return result

    osm_projected = osm_gdf.to_crs(config.utm_crs)
    # Fix invalid geometries
    osm_projected["geometry"] = osm_projected.geometry.apply(
        lambda g: make_valid(g) if not g.is_valid else g
    )

    # Drop rows with missing lat/lon before creating GeoDataFrame
    fsq_valid = fsq_df.dropna(subset=["latitude", "longitude"]).copy()
    fsq_gdf = gpd.GeoDataFrame(
        fsq_valid,
        geometry=gpd.points_from_xy(fsq_valid["longitude"], fsq_valid["latitude"]),
        crs="EPSG:4326",
    )
    fsq_projected = fsq_gdf.to_crs(config.utm_crs)

    records = []
    matched_fsq_ids = set()

    total = len(osm_gdf)
    for i, (idx, osm_row) in enumerate(osm_gdf.iterrows()):
        osm_geom = osm_projected.geometry.iloc[osm_gdf.index.get_loc(idx)]
        area = osm_geom.area
        best_match = None
        best_distance = float("inf")
        best_method = "none"

        for fidx, fsq_row in fsq_projected.iterrows():
            fsq_point = fsq_row.geometry
            if osm_geom.contains(fsq_point):
                distance = 0.0
                if distance < best_distance:
                    best_match = fsq_row
                    best_distance = distance
                    best_method = "contained"
            else:
                distance = osm_geom.distance(fsq_point)
                if distance <= 500 and distance < best_distance:
                    best_match = fsq_row
                    best_distance = distance
                    best_method = "proximity"

        if best_match is not None:
            matched_fsq_ids.add(best_match["fsq_place_id"])
            records.append(
                {
                    "osm_name": osm_row.get("name", "Unknown"),
                    "osm_id": osm_row.get("osm_id", ""),
                    "area_sq_meters": round(area, 2),
                    "area_acres": round(area * 0.000247105, 2),
                    "foursquare_match": True,
                    "foursquare_name": best_match["name"],
                    "foursquare_fsq_place_id": best_match["fsq_place_id"],
                    "distance_meters": round(best_distance, 2),
                    "match_method": best_method,
                }
            )
        else:
            records.append(
                {
                    "osm_name": osm_row.get("name", "Unknown"),
                    "osm_id": osm_row.get("osm_id", ""),
                    "area_sq_meters": round(area, 2),
                    "area_acres": round(area * 0.000247105, 2),
                    "foursquare_match": False,
                    "foursquare_name": "",
                    "foursquare_fsq_place_id": "",
                    "distance_meters": None,
                    "match_method": "none",
                }
            )

        if (i + 1) % 50 == 0:
            print(f"  Validated {i + 1}/{total} OSM polygons...")

    # Add Foursquare-only entries (with lat/lon for map rendering)
    # But first check if any "unmatched" FSQ points actually fall inside an OSM polygon
    # (they may not have been the *best* match for that polygon but still intersect one)
    from shapely.strtree import STRtree
    osm_geoms = list(osm_projected.geometry)
    osm_tree = STRtree(osm_geoms)

    reclassified_count = 0
    for fidx, fsq_row in fsq_valid.iterrows():
        if fsq_row["fsq_place_id"] not in matched_fsq_ids:
            fsq_point = fsq_projected.geometry.loc[fidx]
            # Check if this point falls inside any OSM polygon
            candidate_idxs = osm_tree.query(fsq_point)
            containing_osm = None
            for cidx in candidate_idxs:
                if osm_geoms[cidx].contains(fsq_point):
                    containing_osm = cidx
                    break

            if containing_osm is not None:
                # Reclassify as verified — find the OSM row
                osm_idx = osm_gdf.index[containing_osm]
                osm_row = osm_gdf.loc[osm_idx]
                osm_geom = osm_geoms[containing_osm]
                matched_fsq_ids.add(fsq_row["fsq_place_id"])
                reclassified_count += 1
                # Check if this OSM polygon already has a record; if so, skip duplicate
                osm_id_val = osm_row.get("osm_id", "")
                already_matched = any(
                    r["osm_id"] == osm_id_val and r["foursquare_match"]
                    for r in records
                )
                if not already_matched:
                    records.append(
                        {
                            "osm_name": osm_row.get("name", "Unknown"),
                            "osm_id": osm_id_val,
                            "area_sq_meters": round(osm_geom.area, 2),
                            "area_acres": round(osm_geom.area * 0.000247105, 2),
                            "foursquare_match": True,
                            "foursquare_name": fsq_row["name"],
                            "foursquare_fsq_place_id": fsq_row["fsq_place_id"],
                            "distance_meters": 0.0,
                            "match_method": "contained",
                        }
                    )
                else:
                    # OSM polygon already matched to another FSQ POI; still mark this
                    # FSQ POI as verified (add as additional match record)
                    records.append(
                        {
                            "osm_name": osm_row.get("name", "Unknown"),
                            "osm_id": osm_id_val,
                            "area_sq_meters": round(osm_geom.area, 2),
                            "area_acres": round(osm_geom.area * 0.000247105, 2),
                            "foursquare_match": True,
                            "foursquare_name": fsq_row["name"],
                            "foursquare_fsq_place_id": fsq_row["fsq_place_id"],
                            "distance_meters": 0.0,
                            "match_method": "contained",
                        }
                    )
            else:
                records.append(
                    {
                        "osm_name": "",
                        "osm_id": "",
                        "area_sq_meters": None,
                        "area_acres": None,
                        "foursquare_match": False,
                        "foursquare_name": fsq_row["name"],
                        "foursquare_fsq_place_id": fsq_row["fsq_place_id"],
                        "foursquare_latitude": fsq_row["latitude"],
                        "foursquare_longitude": fsq_row["longitude"],
                        "distance_meters": None,
                        "match_method": "foursquare_only",
                    }
                )

    if reclassified_count > 0:
        print(f"  Reclassified {reclassified_count} Foursquare-only POIs as verified (inside OSM polygons)")

    result = pd.DataFrame(records)
    report_path = os.path.join(DATA_DIR, config.file_path("validation_report.csv"))
    result.to_csv(report_path, index=False)

    matched = result[result["foursquare_match"] == True]
    osm_only = result[
        (result["match_method"] == "none") & (result["osm_id"] != "")
    ]
    fsq_only = result[result["match_method"] == "foursquare_only"]

    print(f"  Matched: {len(matched)}")
    print(f"  OSM-only: {len(osm_only)}")
    print(f"  Foursquare-only: {len(fsq_only)}")
    print(f"  Saved to {report_path}")
    return result












def create_map(gdf, validation_df, config=None):
    """Create an interactive HTML map with folium and a left sidebar."""
    if config is None:
        config = PipelineConfig()
    print("[4/5] Creating interactive map...")

    # Center map on NJ
    gdf_projected = gdf.to_crs(config.utm_crs)
    gdf_centroids = gdf_projected.geometry.centroid.to_crs("EPSG:4326")
    center_lat = gdf_centroids.y.mean()
    center_lon = gdf_centroids.x.mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles=None)

    # Base layers — OpenStreetMap (default), Google Roads, Google Satellite, Mapbox
    folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap").add_to(m)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google", name="Google Roads",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google", name="Google Satellite",
    ).add_to(m)

    if MAPBOX_TOKEN:
        for style, label in [
            ("streets-v12", "Mapbox Streets"),
            ("satellite-streets-v12", "Mapbox Satellite"),
        ]:
            folium.TileLayer(
                tiles=f"https://api.mapbox.com/styles/v1/mapbox/{style}/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}",
                attr="Mapbox", name=label, tileSize=512, zoomOffset=-1,
            ).add_to(m)
    else:
        print("  Note: Set MAPBOX_ACCESS_TOKEN env var to enable Mapbox layers")

    # Determine verified OSM IDs
    fsq_matched_ids = set()
    if not validation_df.empty and "foursquare_match" in validation_df.columns:
        fsq_matched_ids = set(
            validation_df[validation_df["foursquare_match"] == True]["osm_id"].values
        )

    # Create FeatureGroups for toggleable overlay categories
    verified_group = folium.FeatureGroup(name="Verified (OSM + Foursquare)", show=True)
    osm_only_group = folium.FeatureGroup(name="OSM Only", show=True)
    fsq_only_group = folium.FeatureGroup(name="Foursquare Only", show=False)

    # Collect facility data for sidebar
    # Each entry: {name, lat, lon, category}
    facilities = []

    # Add OSM polygons to appropriate groups
    for idx, row in gdf.iterrows():
        osm_id = row.get("osm_id", "")
        name = row.get("name", "Unknown")
        geom_projected = gdf_projected.geometry.iloc[gdf.index.get_loc(idx)]
        area_sqm = geom_projected.area
        area_acres = area_sqm * 0.000247105
        centroid_4326 = gdf_centroids.iloc[gdf.index.get_loc(idx)]

        is_verified = osm_id in fsq_matched_ids
        color = "#2563eb" if is_verified else "#dc2626"  # blue / red
        category = "verified" if is_verified else "osm_only"

        popup_html = f"<b>{name}</b><br>"
        popup_html += f"Area: {area_sqm:,.0f} sq m ({area_acres:,.1f} acres)<br>"
        popup_html += f"OSM ID: {osm_id}<br>"
        popup_html += f"Foursquare: {'Verified' if is_verified else 'Not matched'}<br>"

        geojson_data = row.geometry.__geo_interface__
        layer = folium.GeoJson(
            geojson_data,
            style_function=lambda x, c=color: {
                "fillColor": c, "color": c, "weight": 2, "fillOpacity": 0.3,
            },
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=name,
        )

        if is_verified:
            layer.add_to(verified_group)
        else:
            layer.add_to(osm_only_group)

        facilities.append({
            "name": name,
            "lat": round(centroid_4326.y, 6),
            "lon": round(centroid_4326.x, 6),
            "category": category,
        })

    # Add Foursquare-only points as red markers
    fsq_count = 0
    if not validation_df.empty:
        fsq_only = validation_df[validation_df["match_method"] == "foursquare_only"]
        for _, row in fsq_only.iterrows():
            lat = row.get("foursquare_latitude")
            lon = row.get("foursquare_longitude")
            if not (pd.notna(lat) and pd.notna(lon)):
                continue

            fsq_name = row.get("foursquare_name", "Unknown")

            popup_html = f"<b>{fsq_name}</b><br>"
            popup_html += "Source: Foursquare only (no OSM polygon)<br>"
            popup_html += f"FSQ ID: {row.get('foursquare_fsq_place_id', '')}<br>"

            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="#dc2626",
                fill=True,
                fillColor="#dc2626",
                fillOpacity=0.7,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=fsq_name,
            ).add_to(fsq_only_group)
            fsq_count += 1

            facilities.append({
                "name": fsq_name,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "category": "fsq_only",
            })

    print(f"  Foursquare-only markers: {fsq_count}")

    # Add feature groups to map
    verified_group.add_to(m)
    osm_only_group.add_to(m)
    fsq_only_group.add_to(m)

    # Remove default layer control (we build our own sidebar)
    # We still need folium's layer control JS to toggle layers
    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    # Build facility JSON for sidebar
    import json as _json
    facilities_json = _json.dumps(facilities)

    # Sidebar HTML/CSS/JS as a MacroElement
    # Build the template string with proper Jinja2 syntax
    sidebar_tpl = (
        '{% macro html(this, kwargs) %}\n'
        '<style>\n'
        '#sidebar { position:fixed; top:0; left:0; width:320px; height:100vh; background:#fff;\n'
        '  z-index:1001; box-shadow:2px 0 8px rgba(0,0,0,0.2); display:flex; flex-direction:column;\n'
        '  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; font-size:13px; }\n'
        '#sidebar-header { padding:12px 14px 8px; border-bottom:1px solid #e0e0e0; flex-shrink:0; }\n'
        '#sidebar-header h3 { margin:0 0 10px 0; font-size:15px; color:#333; }\n'
        '.toggle-row { display:flex; align-items:center; margin:4px 0; cursor:pointer; padding:3px 0; }\n'
        '.toggle-row input[type=checkbox] { margin-right:8px; cursor:pointer; }\n'
        '.color-swatch { display:inline-block; width:14px; height:14px; margin-right:6px;\n'
        '  border-radius:2px; vertical-align:middle; }\n'
        '.color-swatch.circle { border-radius:50%; }\n'
        '#search-box { width:100%; padding:7px 10px; margin-top:8px; border:1px solid #ccc;\n'
        '  border-radius:4px; font-size:13px; box-sizing:border-box; }\n'
        '#search-box:focus { outline:none; border-color:#2563eb; box-shadow:0 0 0 2px rgba(37,99,235,0.15); }\n'
        '#facility-list { flex:1; overflow-y:auto; padding:0; margin:0; }\n'
        '.facility-item { padding:6px 14px; cursor:pointer; border-bottom:1px solid #f0f0f0;\n'
        '  display:flex; align-items:center; }\n'
        '.facility-item:hover { background:#f0f7ff; }\n'
        '.facility-item .dot { width:10px; height:10px; border-radius:50%; margin-right:8px; flex-shrink:0; }\n'
        '.facility-item .fname { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }\n'
        '.facility-count { padding:6px 14px; color:#888; font-size:12px;\n'
        '  border-bottom:1px solid #e0e0e0; flex-shrink:0; }\n'
        '.folium-map { margin-left:320px !important; width:calc(100% - 320px) !important; }\n'
        '</style>\n'
        '<div id="sidebar">\n'
        '  <div id="sidebar-header">\n'
        '    <h3>' + config.display_name + '</h3>\n'
        '    <label class="toggle-row">\n'
        '      <input type="checkbox" id="tog-verified" checked onchange="toggleCategory(\'verified\',this.checked)">\n'
        '      <span class="color-swatch" style="background:#2563eb;opacity:0.6;"></span>\n'
        '      Verified (OSM + Foursquare)\n'
        '    </label>\n'
        '    <label class="toggle-row">\n'
        '      <input type="checkbox" id="tog-osm_only" checked onchange="toggleCategory(\'osm_only\',this.checked)">\n'
        '      <span class="color-swatch" style="background:#dc2626;opacity:0.6;"></span>\n'
        '      OSM Only\n'
        '    </label>\n'
        '    <label class="toggle-row">\n'
        '      <input type="checkbox" id="tog-fsq_only" onchange="toggleCategory(\'fsq_only\',this.checked)">\n'
        '      <span class="color-swatch circle" style="background:#dc2626;opacity:0.6;"></span>\n'
        '      Foursquare Only\n'
        '    </label>\n'
        '    <input type="text" id="search-box" placeholder="Search ' + config.keyword.lower() + ' name..." oninput="filterFacilities()">\n'
        '  </div>\n'
        '  <div class="facility-count" id="facility-count"></div>\n'
        '  <div id="facility-list"></div>\n'
        '</div>\n'
        '<script>\n'
        '(function() {\n'
        '  var facilities = FACILITIES_JSON;\n'
        '  var activeCategories = {verified: true, osm_only: true, fsq_only: false};\n'
        '  var mapObj = null;\n'
        '  function getMap() {\n'
        '    if (mapObj) return mapObj;\n'
        '    for (var k in window) {\n'
        '      try { if (window[k] && window[k]._leaflet_id && window[k].getCenter) { mapObj = window[k]; return mapObj; } } catch(e) {}\n'
        '    }\n'
        '    return null;\n'
        '  }\n'
        '  var layerGroups = {};\n'
        '  function findLayerGroups() {\n'
        '    var m = getMap(); if (!m) return;\n'
        '    m.eachLayer(function(layer) {\n'
        '      if (layer.options && layer.options.name) {\n'
        '        var n = layer.options.name;\n'
        '        if (n.indexOf("Verified") >= 0) layerGroups["verified"] = layer;\n'
        '        else if (n.indexOf("OSM Only") >= 0) layerGroups["osm_only"] = layer;\n'
        '        else if (n.indexOf("Foursquare Only") >= 0) layerGroups["fsq_only"] = layer;\n'
        '      }\n'
        '    });\n'
        '  }\n'
        '  window.toggleCategory = function(cat, on) {\n'
        '    activeCategories[cat] = on;\n'
        '    var m = getMap();\n'
        '    if (Object.keys(layerGroups).length === 0) findLayerGroups();\n'
        '    var lg = layerGroups[cat];\n'
        '    if (lg && m) { if (on) m.addLayer(lg); else m.removeLayer(lg); }\n'
        '    renderList();\n'
        '  };\n'
        '  window.filterFacilities = function() { renderList(); };\n'
        '  function renderList() {\n'
        '    var query = document.getElementById("search-box").value.toLowerCase();\n'
        '    var list = document.getElementById("facility-list");\n'
        '    var count = document.getElementById("facility-count");\n'
        '    var html = ""; var n = 0;\n'
        '    var sorted = facilities.slice().sort(function(a,b) { return a.name.localeCompare(b.name); });\n'
        '    for (var i = 0; i < sorted.length; i++) {\n'
        '      var f = sorted[i];\n'
        '      if (!activeCategories[f.category]) continue;\n'
        '      if (query && f.name.toLowerCase().indexOf(query) < 0) continue;\n'
        '      var dotColor = (f.category === "verified") ? "#2563eb" : "#dc2626";\n'
        '      html += \'<div class="facility-item" onmouseenter="panTo(\' + f.lat + \',\' + f.lon + \')" onclick="panTo(\' + f.lat + \',\' + f.lon + \',14)">\';\n'
        '      html += \'<span class="dot" style="background:\' + dotColor + \'"></span>\';\n'
        '      html += \'<span class="fname">\' + f.name + \'</span></div>\';\n'
        '      n++;\n'
        '    }\n'
        '    list.innerHTML = html;\n'
        '    count.textContent = n + " facilities shown";\n'
        '    if (query && n === 1) {\n'
        '      var match = sorted.find(function(f) {\n'
        '        return activeCategories[f.category] && f.name.toLowerCase().indexOf(query) >= 0;\n'
        '      });\n'
        '      if (match) panTo(match.lat, match.lon, 14);\n'
        '    }\n'
        '  }\n'
        '  window.panTo = function(lat, lon, zoom) {\n'
        '    var m = getMap();\n'
        '    if (m) { if (zoom) m.setView([lat, lon], zoom); else m.panTo([lat, lon]); }\n'
        '  };\n'
        '  setTimeout(function() {\n'
        '    findLayerGroups();\n'
        '    var m = getMap();\n'
        '    if (m) {\n'
        '      // osm_only starts visible, no need to remove\n'
        '      if (layerGroups["fsq_only"]) m.removeLayer(layerGroups["fsq_only"]);\n'
        '    }\n'
        '    renderList();\n'
        '  }, 500);\n'
        '})();\n'
        '</script>\n'
        '{% endmacro %}'
    ).replace("FACILITIES_JSON", facilities_json)

    sidebar = MacroElement()
    sidebar._template = Template(sidebar_tpl)
    m.get_root().add_child(sidebar)

    map_path = os.path.join(DATA_DIR, config.file_path("map.html"))
    m.save(map_path)
    print(f"  Map saved to {map_path}")
    return m


def create_editable_map(gdf, validation_df, config=None, fsq_df=None):
    """Create an editable HTML map for reviewing/exporting POI polygons.

    Users can check/uncheck entries, edit polygon vertices via Leaflet-Geoman,
    expand Foursquare-only hexagons into full polygons, and export checked
    polygons as a GeoJSON download.
    """
    if config is None:
        config = PipelineConfig()
    print(f"[Editable Map] Creating editable {config.keyword.lower()} map...")

    # --- Read user guide markdown and convert to simple HTML ---
    guide_html = ""
    guide_md_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "EDITABLE_MAP_USER_GUIDE.md"
    )
    if os.path.exists(guide_md_path):
        with open(guide_md_path, "r") as gf:
            md_text = gf.read()
        html_lines = []
        in_table = False
        in_list = False
        for line in md_text.split("\n"):
            stripped = line.strip()
            if stripped == "---":
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                html_lines.append("<hr>")
                continue
            if stripped.startswith("|") and stripped.endswith("|"):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if all(set(c) <= {"-", " ", ":"} for c in cells):
                    continue
                if not in_table:
                    html_lines.append('<table style="width:100%;border-collapse:collapse;margin:10px 0;">')
                    in_table = True
                    html_lines.append("<tr>" + "".join(
                        '<th style="text-align:left;border-bottom:2px solid #ccc;padding:6px 8px;">' + c + '</th>'
                        for c in cells
                    ) + "</tr>")
                else:
                    html_lines.append("<tr>" + "".join(
                        '<td style="border-bottom:1px solid #eee;padding:6px 8px;">' + c + '</td>'
                        for c in cells
                    ) + "</tr>")
                continue
            if in_table and not stripped.startswith("|"):
                html_lines.append("</table>")
                in_table = False
            if stripped.startswith("### "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append("<h4>" + stripped[4:] + "</h4>")
                continue
            if stripped.startswith("## "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append("<h3>" + stripped[3:] + "</h3>")
                continue
            if stripped.startswith("# "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append("<h2>" + stripped[2:] + "</h2>")
                continue
            if stripped.startswith("- ") or stripped.startswith("* "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                content = stripped[2:]
                content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
                html_lines.append("<li>" + content + "</li>")
                continue
            if stripped and stripped[0].isdigit() and ". " in stripped:
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                content = stripped.split(". ", 1)[1]
                content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
                html_lines.append("<li>" + content + "</li>")
                continue
            if in_list and not stripped:
                html_lines.append("</ul>")
                in_list = False
            if stripped:
                text = stripped
                text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
                text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
                html_lines.append("<p>" + text + "</p>")
        if in_list:
            html_lines.append("</ul>")
        if in_table:
            html_lines.append("</table>")
        guide_html = "\n".join(html_lines)

    # --- 1. Base map setup ---
    gdf_projected = gdf.to_crs(config.utm_crs)
    gdf_centroids = gdf_projected.geometry.centroid.to_crs("EPSG:4326")
    center_lat = gdf_centroids.y.mean()
    center_lon = gdf_centroids.x.mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles=None)

    folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google", name="Google Roads",
    ).add_to(m)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google", name="Google Satellite",
    ).add_to(m)
    if MAPBOX_TOKEN:
        for style, label in [
            ("streets-v12", "Mapbox Streets"),
            ("satellite-streets-v12", "Mapbox Satellite"),
        ]:
            folium.TileLayer(
                tiles=f"https://api.mapbox.com/styles/v1/mapbox/{style}/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}",
                attr="Mapbox", name=label, tileSize=512, zoomOffset=-1,
            ).add_to(m)

    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    v_total = sum(_count_polygon_vertices(g.__geo_interface__) for g in gdf.geometry)
    print(f"  Total vertices (original): {v_total:,}")

    # --- 2. Data preparation ---
    fsq_matched_ids = set()
    if not validation_df.empty and "foursquare_match" in validation_df.columns:
        fsq_matched_ids = set(
            validation_df[validation_df["foursquare_match"] == True]["osm_id"].values
        )

    facilities = []
    fid = 0

    # OSM polygons (Verified + OSM Only)
    for idx, row in gdf.iterrows():
        osm_id = row.get("osm_id", "")
        name = row.get("name", "Unknown")
        is_verified = osm_id in fsq_matched_ids
        category = "verified" if is_verified else "osm_only"

        geojson_geom = row.geometry.__geo_interface__

        # Look up fsq_id + POI coordinates + address for verified courses
        fsq_id = ""
        fsq_name = ""
        fsq_address = ""
        poi_lat = None
        poi_lon = None
        if is_verified and not validation_df.empty:
            match = validation_df[
                (validation_df["osm_id"] == osm_id)
                & (validation_df["foursquare_match"] == True)
            ]
            if not match.empty:
                fsq_id = str(match.iloc[0].get("foursquare_fsq_place_id", ""))
                fsq_name = str(match.iloc[0].get("foursquare_name", ""))
                if fsq_df is not None and not fsq_df.empty and fsq_id:
                    fsq_match = fsq_df[fsq_df["fsq_place_id"] == fsq_id]
                    if not fsq_match.empty:
                        poi_lat = float(fsq_match.iloc[0]["latitude"])
                        poi_lon = float(fsq_match.iloc[0]["longitude"])
                        addr = fsq_match.iloc[0].get("address", "")
                        loc = fsq_match.iloc[0].get("locality", "")
                        parts = [str(p) for p in [addr, loc] if pd.notna(p) and str(p).strip()]
                        fsq_address = ", ".join(parts)

        # Preserve OSM node IDs (parallel to coordinate rings)
        node_ids = row.get("_node_ids", None)
        if node_ids is not None and not isinstance(node_ids, list):
            try:
                node_ids = json.loads(node_ids) if isinstance(node_ids, str) else None
            except (json.JSONDecodeError, TypeError):
                node_ids = None

        facilities.append({
            "id": fid,
            "name": name,
            "category": category,
            "checked": is_verified,
            "geometry": geojson_geom,
            "osm_id": osm_id,
            "fsq_id": fsq_id,
            "fsq_name": fsq_name,
            "fsq_address": fsq_address,
            "poi_lat": poi_lat,
            "poi_lon": poi_lon,
            "_node_ids": node_ids,
        })
        fid += 1

    # Foursquare-only entries → small hexagons
    if not validation_df.empty:
        fsq_only = validation_df[validation_df["match_method"] == "foursquare_only"]
        for _, row in fsq_only.iterrows():
            lat = row.get("foursquare_latitude")
            lon = row.get("foursquare_longitude")
            if not (pd.notna(lat) and pd.notna(lon)):
                continue
            fsq_name = row.get("foursquare_name", "Unknown")
            fsq_place_id = str(row.get("foursquare_fsq_place_id", ""))
            # Look up address from fsq_df
            fsq_addr = ""
            if fsq_df is not None and not fsq_df.empty and fsq_place_id:
                fsq_match = fsq_df[fsq_df["fsq_place_id"] == fsq_place_id]
                if not fsq_match.empty:
                    addr = fsq_match.iloc[0].get("address", "")
                    loc = fsq_match.iloc[0].get("locality", "")
                    parts = [str(p) for p in [addr, loc] if pd.notna(p) and str(p).strip()]
                    fsq_addr = ", ".join(parts)
            hex_geom = _make_hexagon(float(lat), float(lon), radius_m=75)
            facilities.append({
                "id": fid,
                "name": fsq_name,
                "category": "fsq_only",
                "checked": False,
                "geometry": hex_geom,
                "osm_id": "",
                "fsq_id": fsq_place_id,
                "fsq_name": fsq_name,
                "fsq_address": fsq_addr,
                "poi_lat": float(lat),
                "poi_lon": float(lon),
            })
            fid += 1

    facilities_json = json.dumps(facilities)

    n_verified = sum(1 for f in facilities if f["category"] == "verified")
    n_osm = sum(1 for f in facilities if f["category"] == "osm_only")
    n_fsq = sum(1 for f in facilities if f["category"] == "fsq_only")
    print(f"  Facilities: {n_verified} verified, {n_osm} OSM-only, {n_fsq} FSQ-only")

    # Compute bounds for the "Add Course" coordinate check
    _bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    _margin = 0.5
    bounds_min_lat = round(_bounds[1] - _margin, 1)
    bounds_max_lat = round(_bounds[3] + _margin, 1)
    bounds_min_lon = round(_bounds[0] - _margin, 1)
    bounds_max_lon = round(_bounds[2] + _margin, 1)

    # --- 3-6. Full HTML page with Leaflet-Geoman, sidebar, export ---
    # Visibility: all categories visible on map by default (category toggles
    #   control map visibility for decluttering).
    # Checkboxes: export-only (checked = will be included in GeoJSON export).
    #   Verified default checked; OSM Only & FSQ Only default unchecked.
    # Editing: right-click a polygon to enter edit mode (vertices become
    #   draggable). Right-click again, press Escape, or click the floating
    #   "Done Editing" button to finish.
    editable_tpl = (
        '{% macro html(this, kwargs) %}\n'
        '<link rel="stylesheet" href="https://unpkg.com/@geoman-io/leaflet-geoman-free@2.17.0/dist/leaflet-geoman.css" />\n'
        '<script src="https://unpkg.com/@geoman-io/leaflet-geoman-free@2.17.0/dist/leaflet-geoman.min.js"></script>\n'
        '<script src="https://unpkg.com/@turf/turf@7/turf.min.js"></script>\n'
        '<style>\n'
        '#ed-sidebar {\n'
        '  position: fixed; top: 0; left: 0; width: 340px; height: 100vh;\n'
        '  background: #fff; z-index: 1001;\n'
        '  box-shadow: 2px 0 8px rgba(0,0,0,0.2);\n'
        '  display: flex; flex-direction: column;\n'
        '  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;\n'
        '  font-size: 13px;\n'
        '}\n'
        '#ed-sidebar-header {\n'
        '  padding: 12px 14px 8px; border-bottom: 1px solid #e0e0e0; flex-shrink: 0;\n'
        '}\n'
        '#ed-sidebar-header h3 { margin: 0 0 6px 0; font-size: 15px; color: #333; }\n'
        '#ed-sidebar-header .ed-hint { font-size: 11px; color: #888; margin-bottom: 8px; }\n'
        '.ed-cat-section { margin-bottom: 2px; }\n'
        '.ed-cat-header {\n'
        '  display: flex; align-items: center; padding: 3px 0; cursor: pointer;\n'
        '  font-weight: 600; font-size: 12px; color: #555;\n'
        '}\n'
        '.ed-cat-header input[type=checkbox] { display: none; }\n'
        '.ed-toggle {\n'
        '  position: relative; width: 32px; height: 18px; background: #ccc;\n'
        '  border-radius: 9px; margin-right: 8px; flex-shrink: 0;\n'
        '  transition: background 0.2s; cursor: pointer;\n'
        '}\n'
        '.ed-toggle::after {\n'
        '  content: ""; position: absolute; top: 2px; left: 2px;\n'
        '  width: 14px; height: 14px; background: #fff; border-radius: 50%;\n'
        '  transition: transform 0.2s;\n'
        '}\n'
        '.ed-cat-header input:checked + .ed-toggle { background: #4f46e5; }\n'
        '.ed-cat-header input:checked + .ed-toggle::after { transform: translateX(14px); }\n'
        '.ed-swatch {\n'
        '  display: inline-block; width: 12px; height: 12px; margin-right: 6px;\n'
        '  border-radius: 2px; vertical-align: middle;\n'
        '}\n'
        '#ed-search {\n'
        '  width: 100%; padding: 7px 10px; margin-top: 8px; border: 1px solid #ccc;\n'
        '  border-radius: 4px; font-size: 13px; box-sizing: border-box;\n'
        '}\n'
        '#ed-search:focus { outline: none; border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37,99,235,0.15); }\n'
        '.ed-count {\n'
        '  padding: 6px 14px; color: #888; font-size: 12px;\n'
        '  border-bottom: 1px solid #e0e0e0; flex-shrink: 0;\n'
        '}\n'
        '#ed-facility-list { flex: 1; overflow-y: auto; padding: 0; margin: 0; }\n'
        '.ed-item {\n'
        '  padding: 5px 14px; display: flex; align-items: center;\n'
        '  border-bottom: 1px solid #f0f0f0; font-size: 12px;\n'
        '}\n'
        '.ed-item:hover { background: #f0f7ff; }\n'
        '.ed-item input[type=checkbox] { margin-right: 6px; cursor: pointer; flex-shrink: 0; }\n'
        '.ed-item .ed-dot {\n'
        '  width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; flex-shrink: 0;\n'
        '}\n'
        '.ed-item .ed-fname {\n'
        '  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: pointer; flex: 1;\n'
        '}\n'
        '#ed-export-btn {\n'
        '  margin: 10px 14px; padding: 10px; background: #2563eb; color: #fff;\n'
        '  border: none; border-radius: 4px; cursor: pointer; font-size: 13px;\n'
        '  font-weight: 600; flex-shrink: 0;\n'
        '}\n'
        '#ed-export-btn:hover { background: #1d4ed8; }\n'
        '#ed-import-btn {\n'
        '  margin: 0 14px 10px; padding: 10px; background: #fff; color: #2563eb;\n'
        '  border: 2px solid #2563eb; border-radius: 4px; cursor: pointer; font-size: 13px;\n'
        '  font-weight: 600; flex-shrink: 0;\n'
        '}\n'
        '#ed-import-btn:hover { background: #eff6ff; }\n'
        '.folium-map { margin-left: 340px !important; width: calc(100% - 340px) !important; }\n'
        '#ed-edit-bar {\n'
        '  display: none; position: fixed; top: 10px; left: 360px; z-index: 1002;\n'
        '  background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px;\n'
        '  padding: 8px 14px; font-family: -apple-system, sans-serif; font-size: 13px;\n'
        '  box-shadow: 0 2px 8px rgba(0,0,0,0.15); display: none;\n'
        '  align-items: center; gap: 10px;\n'
        '}\n'
        '#ed-edit-bar .ed-edit-name { font-weight: 600; }\n'
        '#ed-edit-bar .ed-done-btn {\n'
        '  background: #f59e0b; color: #fff; border: none; border-radius: 3px;\n'
        '  padding: 4px 12px; cursor: pointer; font-size: 12px; font-weight: 600;\n'
        '}\n'
        '#ed-edit-bar .ed-done-btn:hover { background: #d97706; }\n'
        '#ed-ctx-menu {\n'
        '  display: none; position: fixed; z-index: 2000;\n'
        '  background: #fff; border: 1px solid #ccc; border-radius: 4px;\n'
        '  box-shadow: 0 2px 8px rgba(0,0,0,0.18); min-width: 140px;\n'
        '  font-family: -apple-system, sans-serif; font-size: 13px;\n'
        '}\n'
        '#ed-ctx-menu div {\n'
        '  padding: 8px 14px; cursor: pointer;\n'
        '}\n'
        '#ed-ctx-menu div:hover { background: #f0f7ff; }\n'
        '#ed-ctx-menu div.ed-ctx-danger:hover { background: #fef2f2; color: #dc2626; }\n'
        '#ed-add-course-btn {\n'
        '  margin: 10px 14px 0; padding: 10px; background: #7c3aed; color: #fff;\n'
        '  border: none; border-radius: 4px; cursor: pointer; font-size: 13px;\n'
        '  font-weight: 600; flex-shrink: 0;\n'
        '}\n'
        '#ed-add-course-btn:hover { background: #6d28d9; }\n'
        '#ed-csv-btn {\n'
        '  margin: 6px 14px 0; padding: 10px; background: #16a34a; color: #fff;\n'
        '  border: none; border-radius: 4px; cursor: pointer; font-size: 13px;\n'
        '  font-weight: 600; flex-shrink: 0;\n'
        '}\n'
        '#ed-csv-btn:hover { background: #15803d; }\n'
        '.ed-modal-overlay {\n'
        '  display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;\n'
        '  background: rgba(0,0,0,0.5); z-index: 3000; justify-content: center; align-items: center;\n'
        '}\n'
        '.ed-modal-overlay.ed-show { display: flex; }\n'
        '.ed-modal {\n'
        '  background: #fff; border-radius: 8px; padding: 24px; min-width: 360px;\n'
        '  max-width: 500px; box-shadow: 0 8px 30px rgba(0,0,0,0.3);\n'
        '  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;\n'
        '  font-size: 13px;\n'
        '}\n'
        '.ed-modal h3 { margin: 0 0 16px; font-size: 16px; color: #333; }\n'
        '.ed-modal label { display: block; margin-bottom: 4px; font-weight: 600; color: #555; font-size: 12px; }\n'
        '.ed-modal input[type=text], .ed-modal input[type=number] {\n'
        '  width: 100%; padding: 8px 10px; border: 1px solid #ccc; border-radius: 4px;\n'
        '  font-size: 13px; box-sizing: border-box; margin-bottom: 12px;\n'
        '}\n'
        '.ed-modal input:focus { outline: none; border-color: #7c3aed; box-shadow: 0 0 0 2px rgba(124,58,237,0.15); }\n'
        '.ed-modal-btns { display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px; }\n'
        '.ed-modal-btns button {\n'
        '  padding: 8px 18px; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: 600;\n'
        '}\n'
        '.ed-modal-btns .ed-btn-primary { background: #7c3aed; color: #fff; border: none; }\n'
        '.ed-modal-btns .ed-btn-primary:hover { background: #6d28d9; }\n'
        '.ed-modal-btns .ed-btn-cancel { background: #fff; color: #555; border: 1px solid #ccc; }\n'
        '.ed-modal-btns .ed-btn-cancel:hover { background: #f5f5f5; }\n'
        '.ed-modal-warn { color: #d97706; font-size: 12px; margin: -8px 0 8px; }\n'
        '#ed-help-btn {\n'
        '  width: 24px; height: 24px; border-radius: 50%; background: #e0e7ff; color: #4338ca;\n'
        '  border: 1px solid #a5b4fc; font-weight: 700; font-size: 14px; cursor: pointer;\n'
        '  display: inline-flex; align-items: center; justify-content: center;\n'
        '  margin-left: 8px; vertical-align: middle; flex-shrink: 0;\n'
        '}\n'
        '#ed-help-btn:hover { background: #c7d2fe; }\n'
        '#ed-help-modal .ed-modal {\n'
        '  max-width: 640px; max-height: 80vh; overflow-y: auto;\n'
        '}\n'
        '#ed-help-modal .ed-modal h2 { font-size: 18px; margin: 16px 0 8px; color: #333; }\n'
        '#ed-help-modal .ed-modal h3 { font-size: 15px; margin: 14px 0 6px; color: #444; }\n'
        '#ed-help-modal .ed-modal h4 { font-size: 13px; margin: 12px 0 4px; color: #555; }\n'
        '#ed-help-modal .ed-modal p { margin: 6px 0; line-height: 1.5; color: #444; }\n'
        '#ed-help-modal .ed-modal ul { margin: 6px 0; padding-left: 20px; }\n'
        '#ed-help-modal .ed-modal li { margin: 3px 0; line-height: 1.5; color: #444; }\n'
        '#ed-help-modal .ed-modal hr { border: none; border-top: 1px solid #e0e0e0; margin: 16px 0; }\n'
        '#ed-help-modal .ed-modal code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px; font-size: 12px; }\n'
        '</style>\n'
        '<div id="ed-ctx-menu">\n'
        '  <div onclick="edCtxRename()">Rename</div>\n'
        '  <div id="ed-ctx-verify" onclick="edCtxVerify()">Verify</div>\n'
        '  <div onclick="edCtxEdit()">Edit Shape</div>\n'
        '  <div onclick="edCtxSimplify()">Simplify Vertices</div>\n'
        '  <div class="ed-ctx-danger" onclick="edCtxDelete()">Delete</div>\n'
        '</div>\n'
        '<div id="ed-edit-bar">\n'
        '  Editing: <span class="ed-edit-name" id="ed-edit-name"></span>\n'
        '  <span style="color:#92400e;font-size:11px;">(drag vertices to reshape &mdash; right-click or Esc to finish)</span>\n'
        '  <button class="ed-done-btn" onclick="edStopEditing()">Done</button>\n'
        '</div>\n'
        '<!-- Add Course modal -->\n'
        '<div id="ed-add-modal" class="ed-modal-overlay">\n'
        '  <div class="ed-modal">\n'
        '    <h3>Add New ' + config.keyword + '</h3>\n'
        '    <label for="ed-add-name">Name (required)</label>\n'
        '    <input type="text" id="ed-add-name" placeholder="e.g. Pine Valley Golf Club" oninput="edAddCheckDup()">\n'
        '    <div id="ed-add-dup-warn" class="ed-modal-warn" style="display:none;"></div>\n'
        '    <label for="ed-add-lat">Latitude (required)</label>\n'
        '    <input type="number" id="ed-add-lat" step="any" placeholder="e.g. 40.123">\n'
        '    <label for="ed-add-lon">Longitude (required)</label>\n'
        '    <input type="number" id="ed-add-lon" step="any" placeholder="e.g. -74.567">\n'
        '    <div class="ed-modal-btns">\n'
        '      <button class="ed-btn-cancel" onclick="edAddClose()">Cancel</button>\n'
        '      <button class="ed-btn-primary" onclick="edAddSubmit()">Add Course</button>\n'
        '    </div>\n'
        '  </div>\n'
        '</div>\n'
        '<!-- Help modal -->\n'
        '<div id="ed-help-modal" class="ed-modal-overlay">\n'
        '  <div class="ed-modal">\n'
        '    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">\n'
        '      <h3 style="margin:0;">User Guide</h3>\n'
        '      <button class="ed-btn-cancel" onclick="edHelpClose()" style="padding:4px 12px;">Close</button>\n'
        '    </div>\n'
        '    <div id="ed-help-content">GUIDE_HTML_PLACEHOLDER</div>\n'
        '  </div>\n'
        '</div>\n'
        '<div id="ed-sidebar">\n'
        '  <div id="ed-sidebar-header">\n'
        '    <div style="display:flex;align-items:center;">\n'
        '      <h3 style="flex:1;">Editable ' + config.display_name + '</h3>\n'
        '      <button id="ed-help-btn" onclick="edHelpOpen()" title="Help / Instructions">?</button>\n'
        '    </div>\n'
        '    <div class="ed-hint">Toggle = show/hide on map. Checkbox = include in export. Right-click polygon to edit or delete.</div>\n'
        '    <div class="ed-cat-section">\n'
        '      <label class="ed-cat-header">\n'
        '        <input type="checkbox" id="vis-verified" checked onchange="edToggleVis(\'verified\',this.checked)">\n'
        '        <span class="ed-toggle"></span>\n'
        '        <span class="ed-swatch" style="background:#2563eb;opacity:0.7;"></span>\n'
        '        Verified (OSM + Foursquare)\n'
        '      </label>\n'
        '    </div>\n'
        '    <div class="ed-cat-section">\n'
        '      <label class="ed-cat-header">\n'
        '        <input type="checkbox" id="vis-osm_only" checked onchange="edToggleVis(\'osm_only\',this.checked)">\n'
        '        <span class="ed-toggle"></span>\n'
        '        <span class="ed-swatch" style="background:#dc2626;opacity:0.7;"></span>\n'
        '        OSM Only\n'
        '      </label>\n'
        '    </div>\n'
        '    <div class="ed-cat-section">\n'
        '      <label class="ed-cat-header">\n'
        '        <input type="checkbox" id="vis-fsq_only" checked onchange="edToggleVis(\'fsq_only\',this.checked)">\n'
        '        <span class="ed-toggle"></span>\n'
        '        <span class="ed-swatch" style="background:#dc2626;opacity:0.7;"></span>\n'
        '        Foursquare Only\n'
        '      </label>\n'
        '    </div>\n'
        '    <input type="text" id="ed-search" placeholder="Search ' + config.keyword.lower() + ' name..." oninput="edRenderList()">\n'
        '    <label style="display:flex;align-items:center;margin-top:6px;cursor:pointer;user-select:none;">\n'
        '      <input type="checkbox" id="ed-hover-toggle" checked onchange="edSetHoverPan(this.checked)" style="margin-right:6px;cursor:pointer;">\n'
        '      <span style="font-size:12px;color:#555;">Hover to pan (uncheck to require click)</span>\n'
        '    </label>\n'
        '    <div id="ed-buffer-section" style="display:flex;align-items:center;margin-top:6px;gap:8px;">\n'
        '      <label style="display:flex;align-items:center;cursor:pointer;user-select:none;">\n'
        '        <input type="checkbox" id="ed-buffer-toggle" onchange="edToggleBuffers(this.checked)" style="margin-right:6px;cursor:pointer;">\n'
        '        <span style="font-size:12px;color:#555;">Show Buffers</span>\n'
        '      </label>\n'
        '      <label style="display:flex;align-items:center;font-size:12px;color:#555;">\n'
        '        <input type="number" id="ed-buffer-radius" value="1" min="0.1" step="0.1"\n'
        '          style="width:50px;padding:2px 4px;border:1px solid #ccc;border-radius:3px;font-size:12px;text-align:center;"\n'
        '          onchange="edUpdateBufferRadius(+this.value)">\n'
        '        <span style="margin-left:4px;">mi</span>\n'
        '      </label>\n'
        '    </div>\n'
        '  </div>\n'
        '  <div class="ed-count" id="ed-count"></div>\n'
        '  <div id="ed-facility-list"></div>\n'
        '  <button id="ed-add-course-btn" onclick="edAddOpen()">+ Add Course</button>\n'
        '  <button id="ed-export-btn" onclick="edExport()">Export GeoJSON</button>\n'
        '  <button id="ed-csv-btn" onclick="edExportCSV()">Export Checked to CSV</button>\n'
        '  <button id="ed-import-btn" onclick="document.getElementById(\'ed-import-file\').click()">Import GeoJSON</button>\n'
        '  <input type="file" id="ed-import-file" accept=".geojson,.json" style="display:none" onchange="edImport(this)">\n'
        '</div>\n'
        '<script>\n'
        '(function() {\n'
        '  var facilities = FACILITIES_JSON;\n'
        '  var layers = {};       // id -> L.geoJSON layer\n'
        '  var fState = {};       // id -> {checked}\n'
        '  var catVisible = {verified: true, osm_only: true, fsq_only: true};\n'
        '  var editingId = null;  // id of polygon currently being edited\n'
        '  var hoverPanEnabled = true;\n'
        '  var deletedIds = {};   // id -> true for deleted facilities\n'
        '  var ctxTargetId = null;\n'
        '  var hoveredId = null;  // id of polygon currently under mouse\n'
        '  var mapObj = null;\n'
        '  var bufferLayers = {};     // id -> L.geoJSON buffer layer\n'
        '  var bufferVisible = false;\n'
        '  var bufferRadiusMiles = 1;\n'
        '\n'
        '  function getMap() {\n'
        '    if (mapObj) return mapObj;\n'
        '    for (var k in window) {\n'
        '      try {\n'
        '        if (window[k] && window[k]._leaflet_id && window[k].getCenter) {\n'
        '          mapObj = window[k]; return mapObj;\n'
        '        }\n'
        '      } catch(e) {}\n'
        '    }\n'
        '    return null;\n'
        '  }\n'
        '\n'
        '  function facilityById(id) {\n'
        '    for (var i = 0; i < facilities.length; i++) {\n'
        '      if (facilities[i].id === id) return facilities[i];\n'
        '    }\n'
        '    return null;\n'
        '  }\n'
        '\n'
        '  // --- Buffer computation ---\n'
        '  function computeBufferGeom(geom) {\n'
        '    try {\n'
        '      var tf = turf.feature(geom);\n'
        '      var buffered = turf.buffer(tf, bufferRadiusMiles, {units: "miles"});\n'
        '      return buffered ? buffered.geometry : null;\n'
        '    } catch(e) { return null; }\n'
        '  }\n'
        '  function updateBuffer(id) {\n'
        '    var m = getMap(); if (!m) return;\n'
        '    if (bufferLayers[id]) {\n'
        '      if (m.hasLayer(bufferLayers[id])) m.removeLayer(bufferLayers[id]);\n'
        '      delete bufferLayers[id];\n'
        '    }\n'
        '    var geom = null;\n'
        '    var layer = layers[id];\n'
        '    if (layer) { layer.eachLayer(function(sub) { if (sub.toGeoJSON) geom = sub.toGeoJSON().geometry; }); }\n'
        '    if (!geom) { var f = facilityById(id); if (f) geom = f.geometry; }\n'
        '    if (!geom) return;\n'
        '    var bufGeom = computeBufferGeom(geom);\n'
        '    if (!bufGeom) return;\n'
        '    var bl = L.geoJSON({type:"Feature",geometry:bufGeom,properties:{}}, {\n'
        '      style: {fillColor:"#f59e0b",color:"#d97706",weight:1.5,fillOpacity:0.1,dashArray:"6,4"},\n'
        '      interactive: false\n'
        '    });\n'
        '    bufferLayers[id] = bl;\n'
        '    if (bufferVisible && fState[id] && fState[id].checked && !deletedIds[id]) bl.addTo(m);\n'
        '  }\n'
        '  function updateAllBuffers() {\n'
        '    for (var i = 0; i < facilities.length; i++) updateBuffer(facilities[i].id);\n'
        '  }\n'
        '\n'
        '  // --- Editing ---\n'
        '  function startEditing(id) {\n'
        '    if (editingId !== null) stopEditingInternal();\n'
        '    var layer = layers[id];\n'
        '    if (!layer) return;\n'
        '    editingId = id;\n'
        '    layer.eachLayer(function(sub) {\n'
        '      sub.unbindTooltip();\n'
        '      sub.setStyle({weight: 3, dashArray: "6,4"});\n'
        '      if (sub.pm) sub.pm.enable({allowSelfIntersection: false});\n'
        '    });\n'
        '    var f = facilityById(id);\n'
        '    var bar = document.getElementById("ed-edit-bar");\n'
        '    document.getElementById("ed-edit-name").textContent = f ? f.name : "";\n'
        '    bar.style.display = "flex";\n'
        '  }\n'
        '  function stopEditingInternal() {\n'
        '    if (editingId === null) return;\n'
        '    var prevId = editingId;\n'
        '    var layer = layers[editingId];\n'
        '    var f = facilityById(editingId);\n'
        '    if (layer) {\n'
        '      var color = (f && f.category === "verified") ? "#2563eb" : "#dc2626";\n'
        '      layer.eachLayer(function(sub) {\n'
        '        if (sub.pm) sub.pm.disable();\n'
        '        sub.setStyle({weight: 2, dashArray: null, color: color});\n'
        '        if (f) sub.bindTooltip(f.name);\n'
        '      });\n'
        '    }\n'
        '    editingId = null;\n'
        '    document.getElementById("ed-edit-bar").style.display = "none";\n'
        '    updateBuffer(prevId);\n'
        '  }\n'
        '  window.edStopEditing = function() { stopEditingInternal(); };\n'
        '\n'
        '  // --- Context menu ---\n'
        '  function showCtxMenu(x, y, id) {\n'
        '    ctxTargetId = id;\n'
        '    var f = facilityById(id);\n'
        '    var vEl = document.getElementById("ed-ctx-verify");\n'
        '    if (vEl && f) vEl.textContent = (f.category === "verified") ? "Unverify" : "Verify";\n'
        '    var menu = document.getElementById("ed-ctx-menu");\n'
        '    menu.style.left = x + "px";\n'
        '    menu.style.top = y + "px";\n'
        '    menu.style.display = "block";\n'
        '  }\n'
        '  function hideCtxMenu() {\n'
        '    document.getElementById("ed-ctx-menu").style.display = "none";\n'
        '    ctxTargetId = null;\n'
        '  }\n'
        '  document.addEventListener("click", function() { hideCtxMenu(); });\n'
        '  window.edSidebarCtx = function(e, id) {\n'
        '    e.preventDefault();\n'
        '    showCtxMenu(e.clientX, e.clientY, id);\n'
        '  };\n'
        '  window.edCtxEdit = function() {\n'
        '    var id = ctxTargetId;\n'
        '    hideCtxMenu();\n'
        '    if (id === null) return;\n'
        '    if (editingId === id) stopEditingInternal();\n'
        '    else startEditing(id);\n'
        '  };\n'
        '  function deleteFeature(id) {\n'
        '    if (editingId === id) stopEditingInternal();\n'
        '    var m = getMap();\n'
        '    if (m && layers[id] && m.hasLayer(layers[id])) m.removeLayer(layers[id]);\n'
        '    if (m && bufferLayers[id] && m.hasLayer(bufferLayers[id])) m.removeLayer(bufferLayers[id]);\n'
        '    deletedIds[id] = true;\n'
        '    fState[id].checked = false;\n'
        '    edRenderList();\n'
        '  }\n'
        '  window.edCtxDelete = function() {\n'
        '    var id = ctxTargetId;\n'
        '    hideCtxMenu();\n'
        '    if (id === null) return;\n'
        '    deleteFeature(id);\n'
        '  };\n'
        '  window.edCtxVerify = function() {\n'
        '    var id = ctxTargetId;\n'
        '    hideCtxMenu();\n'
        '    if (id === null) return;\n'
        '    var f = facilityById(id);\n'
        '    if (!f) return;\n'
        '    var wasVerified = (f.category === "verified");\n'
        '    f.category = wasVerified ? "osm_only" : "verified";\n'
        '    var newColor = (f.category === "verified") ? "#2563eb" : "#dc2626";\n'
        '    if (layers[id]) {\n'
        '      layers[id].eachLayer(function(sub) {\n'
        '        sub.setStyle({fillColor: newColor, color: newColor});\n'
        '      });\n'
        '    }\n'
        '    if (!wasVerified) { fState[id].checked = true; }\n'
        '    edRenderList();\n'
        '  };\n'
        '\n'
        '  // --- Simplify (Ramer-Douglas-Peucker) ---\n'
        '  function rdpDist(p, a, b) {\n'
        '    var dx = b[0]-a[0], dy = b[1]-a[1];\n'
        '    var len2 = dx*dx + dy*dy;\n'
        '    if (len2 === 0) return Math.sqrt((p[0]-a[0])*(p[0]-a[0])+(p[1]-a[1])*(p[1]-a[1]));\n'
        '    var t = Math.max(0, Math.min(1, ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / len2));\n'
        '    var px = a[0]+t*dx, py = a[1]+t*dy;\n'
        '    return Math.sqrt((p[0]-px)*(p[0]-px)+(p[1]-py)*(p[1]-py));\n'
        '  }\n'
        '  function rdpSimplify(pts, eps) {\n'
        '    if (pts.length <= 2) return pts;\n'
        '    var maxD = 0, idx = 0;\n'
        '    for (var i = 1; i < pts.length - 1; i++) {\n'
        '      var d = rdpDist(pts[i], pts[0], pts[pts.length-1]);\n'
        '      if (d > maxD) { maxD = d; idx = i; }\n'
        '    }\n'
        '    if (maxD > eps) {\n'
        '      var left = rdpSimplify(pts.slice(0, idx+1), eps);\n'
        '      var right = rdpSimplify(pts.slice(idx), eps);\n'
        '      return left.slice(0, left.length-1).concat(right);\n'
        '    }\n'
        '    return [pts[0], pts[pts.length-1]];\n'
        '  }\n'
        '  function simplifyLayer(id) {\n'
        '    if (editingId === id) stopEditingInternal();\n'
        '    var layer = layers[id];\n'
        '    var f = facilityById(id);\n'
        '    if (!layer || !f) return;\n'
        '    var m = getMap();\n'
        '    // Tolerance in degrees (~10m at mid-latitudes)\n'
        '    var eps = 0.0001;\n'
        '    var geom = null;\n'
        '    layer.eachLayer(function(sub) { if (sub.toGeoJSON) geom = sub.toGeoJSON().geometry; });\n'
        '    if (!geom) return;\n'
        '    var before = 0, after = 0;\n'
        '    if (geom.type === "Polygon") {\n'
        '      for (var r = 0; r < geom.coordinates.length; r++) {\n'
        '        before += geom.coordinates[r].length;\n'
        '        geom.coordinates[r] = rdpSimplify(geom.coordinates[r], eps);\n'
        '        after += geom.coordinates[r].length;\n'
        '      }\n'
        '    } else if (geom.type === "MultiPolygon") {\n'
        '      for (var p = 0; p < geom.coordinates.length; p++) {\n'
        '        for (var r = 0; r < geom.coordinates[p].length; r++) {\n'
        '          before += geom.coordinates[p][r].length;\n'
        '          geom.coordinates[p][r] = rdpSimplify(geom.coordinates[p][r], eps);\n'
        '          after += geom.coordinates[p][r].length;\n'
        '        }\n'
        '      }\n'
        '    }\n'
        '    // Replace layer\n'
        '    var wasVisible = m && m.hasLayer(layer);\n'
        '    if (wasVisible) m.removeLayer(layer);\n'
        '    var color = (f.category === "verified") ? "#2563eb" : "#dc2626";\n'
        '    var newFeat = {type:"Feature", geometry: geom, properties:{id:f.id, name:f.name}};\n'
        '    var newLayer = L.geoJSON(newFeat, {\n'
        '      style: {fillColor: color, color: color, weight: 2, fillOpacity: 0.3}\n'
        '    });\n'
        '    newLayer.eachLayer(function(sub) {\n'
        '      sub.bindTooltip(f.name);\n'
        '      var sid = f.id;\n'
        '      sub.on("contextmenu", function(ev) {\n'
        '        L.DomEvent.preventDefault(ev);\n'
        '        showCtxMenu(ev.originalEvent.clientX, ev.originalEvent.clientY, sid);\n'
        '      });\n'
        '      sub.on("mouseover", function() { hoveredId = sid; });\n'
        '      sub.on("mouseout", function() { if (hoveredId === sid) hoveredId = null; });\n'
        '    });\n'
        '    layers[f.id] = newLayer;\n'
        '    if (wasVisible) newLayer.addTo(m);\n'
        '    // Clear node IDs since simplification breaks the OSM node mapping\n'
        '    f._node_ids = null;\n'
        '    alert("Simplified: " + before + " \\u2192 " + after + " vertices");\n'
        '  }\n'
        '  window.edCtxSimplify = function() {\n'
        '    var id = ctxTargetId;\n'
        '    hideCtxMenu();\n'
        '    if (id !== null) simplifyLayer(id);\n'
        '  };\n'
        '  window.edCtxRename = function() {\n'
        '    var id = ctxTargetId;\n'
        '    hideCtxMenu();\n'
        '    if (id === null) return;\n'
        '    var f = facilityById(id);\n'
        '    if (!f) return;\n'
        '    var newName = prompt("Rename course:", f.name);\n'
        '    if (newName === null || newName.trim() === "") return;\n'
        '    newName = newName.trim();\n'
        '    if (newName === f.name) return;\n'
        '    // Check for duplicate names\n'
        '    var dupes = [];\n'
        '    for (var i = 0; i < facilities.length; i++) {\n'
        '      if (facilities[i].id !== id && facilities[i].name.toLowerCase() === newName.toLowerCase() && !deletedIds[facilities[i].id]) {\n'
        '        dupes.push(facilities[i].name);\n'
        '      }\n'
        '    }\n'
        '    if (dupes.length > 0) {\n'
        '      if (!confirm("A course named \\\"" + dupes[0] + "\\\" already exists. Rename anyway?")) return;\n'
        '    }\n'
        '    f.name = newName;\n'
        '    // Update tooltip on map layer\n'
        '    if (layers[id]) {\n'
        '      layers[id].eachLayer(function(sub) {\n'
        '        sub.unbindTooltip();\n'
        '        sub.bindTooltip(newName);\n'
        '      });\n'
        '    }\n'
        '    edRenderList();\n'
        '  };\n'
        '\n'
        '  // --- Layer init ---\n'
        '  function initLayers() {\n'
        '    var m = getMap(); if (!m) return;\n'
        '    for (var i = 0; i < facilities.length; i++) {\n'
        '      (function(f) {\n'
        '        var color = (f.category === "verified") ? "#2563eb" : "#dc2626";\n'
        '        var geojsonFeature = {\n'
        '          type: "Feature",\n'
        '          geometry: f.geometry,\n'
        '          properties: {id: f.id, name: f.name}\n'
        '        };\n'
        '        var layer = L.geoJSON(geojsonFeature, {\n'
        '          style: {fillColor: color, color: color, weight: 2, fillOpacity: 0.3}\n'
        '        });\n'
        '        layer.eachLayer(function(sub) {\n'
        '          sub.bindTooltip(f.name);\n'
        '          sub.on("contextmenu", function(e) {\n'
        '            L.DomEvent.preventDefault(e);\n'
        '            showCtxMenu(e.originalEvent.clientX, e.originalEvent.clientY, f.id);\n'
        '          });\n'
        '          sub.on("mouseover", function() { hoveredId = f.id; });\n'
        '          sub.on("mouseout", function() { if (hoveredId === f.id) hoveredId = null; });\n'
        '        });\n'
        '        layers[f.id] = layer;\n'
        '        fState[f.id] = {checked: f.checked};\n'
        '        // All layers visible on load (category toggles control visibility)\n'
        '        layer.addTo(m);\n'
        '      })(facilities[i]);\n'
        '    }\n'
        '  }\n'
        '\n'
        '  // --- Visibility (category-level) ---\n'
        '  function showLayer(id) {\n'
        '    var m = getMap(); if (!m || !layers[id]) return;\n'
        '    if (!m.hasLayer(layers[id])) layers[id].addTo(m);\n'
        '  }\n'
        '  function hideLayer(id) {\n'
        '    var m = getMap(); if (!m || !layers[id]) return;\n'
        '    if (m.hasLayer(layers[id])) m.removeLayer(layers[id]);\n'
        '  }\n'
        '\n'
        '  window.edToggleVis = function(cat, on) {\n'
        '    catVisible[cat] = on;\n'
        '    for (var i = 0; i < facilities.length; i++) {\n'
        '      var f = facilities[i];\n'
        '      if (f.category !== cat) continue;\n'
        '      if (on) showLayer(f.id); else hideLayer(f.id);\n'
        '    }\n'
        '    edRenderList();\n'
        '  };\n'
        '\n'
        '  // --- Export checkbox (individual) ---\n'
        '  window.edToggleItem = function(id, on) {\n'
        '    fState[id].checked = on;\n'
        '    if (bufferVisible && bufferLayers[id]) {\n'
        '      var m = getMap();\n'
        '      if (m) {\n'
        '        if (on && !deletedIds[id]) { if (!m.hasLayer(bufferLayers[id])) bufferLayers[id].addTo(m); }\n'
        '        else { if (m.hasLayer(bufferLayers[id])) m.removeLayer(bufferLayers[id]); }\n'
        '      }\n'
        '    }\n'
        '    // Update the counter\n'
        '    var checked = 0;\n'
        '    for (var i = 0; i < facilities.length; i++) { if (fState[facilities[i].id].checked) checked++; }\n'
        '    var countEl = document.getElementById("ed-count");\n'
        '    if (countEl) {\n'
        '      var parts = countEl.textContent.split(",");\n'
        '      countEl.textContent = parts[0] + ", " + checked + " validated";\n'
        '    }\n'
        '  };\n'
        '\n'
        '  // --- Hover-to-pan toggle ---\n'
        '  window.edSetHoverPan = function(on) {\n'
        '    hoverPanEnabled = on;\n'
        '    edRenderList();\n'
        '  };\n'
        '\n'
        '  // --- Buffer toggle / radius ---\n'
        '  window.edToggleBuffers = function(on) {\n'
        '    bufferVisible = on;\n'
        '    var m = getMap(); if (!m) return;\n'
        '    for (var id in bufferLayers) {\n'
        '      var nid = +id;\n'
        '      if (on && fState[nid] && fState[nid].checked && !deletedIds[nid]) {\n'
        '        if (!m.hasLayer(bufferLayers[id])) bufferLayers[id].addTo(m);\n'
        '      } else {\n'
        '        if (m.hasLayer(bufferLayers[id])) m.removeLayer(bufferLayers[id]);\n'
        '      }\n'
        '    }\n'
        '  };\n'
        '  window.edUpdateBufferRadius = function(miles) {\n'
        '    if (!miles || miles <= 0) return;\n'
        '    bufferRadiusMiles = miles;\n'
        '    updateAllBuffers();\n'
        '  };\n'
        '\n'
        '  // --- Zoom / Pan ---\n'
        '  window.edZoomTo = function(id) {\n'
        '    var m = getMap(); if (!m || !layers[id]) return;\n'
        '    var bounds = layers[id].getBounds();\n'
        '    if (bounds.isValid()) m.fitBounds(bounds, {maxZoom: 16, padding: [40, 40]});\n'
        '  };\n'
        '  window.edPanTo = function(id) {\n'
        '    var m = getMap(); if (!m || !layers[id]) return;\n'
        '    var bounds = layers[id].getBounds();\n'
        '    if (bounds.isValid()) m.panTo(bounds.getCenter());\n'
        '  };\n'
        '\n'
        '  // --- Sidebar list ---\n'
        '  window.edRenderList = function() {\n'
        '    var query = document.getElementById("ed-search").value.toLowerCase();\n'
        '    var list = document.getElementById("ed-facility-list");\n'
        '    var countEl = document.getElementById("ed-count");\n'
        '    var cats = [\n'
        '      {key: "verified", label: "Verified (OSM + Foursquare)"},\n'
        '      {key: "osm_only", label: "OSM Only"},\n'
        '      {key: "fsq_only", label: "Foursquare Only"}\n'
        '    ];\n'
        '    var html = ""; var totalShown = 0;\n'
        '    for (var c = 0; c < cats.length; c++) {\n'
        '      var cat = cats[c];\n'
        '      if (!catVisible[cat.key]) continue;\n'
        '      var items = facilities.filter(function(f) {\n'
        '        return f.category === cat.key && !deletedIds[f.id] &&\n'
        '               (!query || f.name.toLowerCase().indexOf(query) >= 0);\n'
        '      }).sort(function(a,b){ return a.name.localeCompare(b.name); });\n'
        '      if (items.length === 0) continue;\n'
        '      var dotColor = (cat.key === "verified") ? "#2563eb" : "#dc2626";\n'
        '      html += \'<div style="padding:6px 14px 2px;font-weight:600;font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.5px;">\' + cat.label + \' (\' + items.length + \')</div>\';\n'
        '      for (var j = 0; j < items.length; j++) {\n'
        '        var f = items[j];\n'
        '        var chk = fState[f.id].checked ? " checked" : "";\n'
        '        html += \'<div class="ed-item">\';\n'
        '        html += \'<input type="checkbox" title="Include in export"\' + chk + \' onchange="edToggleItem(\' + f.id + \',this.checked)">\';\n'
        '        html += \'<span class="ed-dot" style="background:\' + dotColor + \'"></span>\';\n'
        '        var hov = hoverPanEnabled ? \' onmouseenter="edPanTo(\' + f.id + \')"\'  : "";\n'
        '        html += \'<span class="ed-fname"\' + hov + \' onclick="edZoomTo(\' + f.id + \')" oncontextmenu="edSidebarCtx(event,\' + f.id + \')">\' + f.name + \'</span>\';\n'
        '        html += \'</div>\';\n'
        '        totalShown++;\n'
        '      }\n'
        '    }\n'
        '    list.innerHTML = html;\n'
        '    var checked = 0;\n'
        '    for (var i = 0; i < facilities.length; i++) { if (fState[facilities[i].id].checked) checked++; }\n'
        '    countEl.textContent = totalShown + " shown, " + checked + " validated";\n'
        '  };\n'
        '\n'
        '  // --- Export ---\n'
        '  window.edExport = function() {\n'
        '    var features = [];\n'
        '    for (var i = 0; i < facilities.length; i++) {\n'
        '      var f = facilities[i];\n'
        '      var isDeleted = !!deletedIds[f.id];\n'
        '      var isChecked = !!fState[f.id].checked;\n'
        '      var geom = null;\n'
        '      var layer = layers[f.id];\n'
        '      if (layer) {\n'
        '        layer.eachLayer(function(sub) {\n'
        '          if (sub.toGeoJSON) geom = sub.toGeoJSON().geometry;\n'
        '        });\n'
        '      }\n'
        '      if (!geom) geom = f.geometry || null;\n'
        '      if (!geom) continue;\n'
        '      features.push({\n'
        '        type: "Feature",\n'
        '        properties: {\n'
        '          name: f.name,\n'
        '          category: f.category,\n'
        '          osm_id: f.osm_id,\n'
        '          fsq_id: f.fsq_id,\n'
        '          _node_ids: f._node_ids || null,\n'
        '          checked: isChecked,\n'
        '          deleted: isDeleted,\n'
        '          poi_lat: (f.poi_lat != null) ? f.poi_lat : null,\n'
        '          poi_lon: (f.poi_lon != null) ? f.poi_lon : null,\n'
        '          fsq_name: f.fsq_name || "",\n'
        '          fsq_address: f.fsq_address || "",\n'
        '          buffer_radius_miles: bufferRadiusMiles\n'
        '        },\n'
        '        geometry: geom\n'
        '      });\n'
        '    }\n'
        '    // Add buffer features for checked non-deleted facilities\n'
        '    for (var i = 0; i < facilities.length; i++) {\n'
        '      var bf = facilities[i];\n'
        '      if (!fState[bf.id].checked || deletedIds[bf.id]) continue;\n'
        '      var bufGeom = null;\n'
        '      if (bufferLayers[bf.id]) {\n'
        '        bufferLayers[bf.id].eachLayer(function(sub) {\n'
        '          if (sub.toGeoJSON) bufGeom = sub.toGeoJSON().geometry;\n'
        '        });\n'
        '      }\n'
        '      if (!bufGeom) {\n'
        '        var srcGeom = null;\n'
        '        if (layers[bf.id]) layers[bf.id].eachLayer(function(sub) { if (sub.toGeoJSON) srcGeom = sub.toGeoJSON().geometry; });\n'
        '        if (srcGeom) bufGeom = computeBufferGeom(srcGeom);\n'
        '      }\n'
        '      if (bufGeom) {\n'
        '        features.push({\n'
        '          type: "Feature",\n'
        '          properties: {\n'
        '            _is_buffer: true,\n'
        '            _buffer_for_name: bf.name,\n'
        '            _buffer_for_osm_id: bf.osm_id || "",\n'
        '            buffer_radius_miles: bufferRadiusMiles\n'
        '          },\n'
        '          geometry: bufGeom\n'
        '        });\n'
        '      }\n'
        '    }\n'
        '    if (features.length === 0) { alert("No facilities to export."); return; }\n'
        '    var fc = {type: "FeatureCollection", features: features, buffer_radius_miles: bufferRadiusMiles};\n'
        '    var blob = new Blob([JSON.stringify(fc, null, 2)], {type: "application/json"});\n'
        '    var url = URL.createObjectURL(blob);\n'
        '    var a = document.createElement("a");\n'
        '    a.href = url; a.download = "EDITED_GEOJSON_FILENAME";\n'
        '    document.body.appendChild(a); a.click();\n'
        '    document.body.removeChild(a); URL.revokeObjectURL(url);\n'
        '    var chk = 0, del = 0;\n'
        '    for (var i = 0; i < features.length; i++) {\n'
        '      if (features[i].properties.checked) chk++;\n'
        '      if (features[i].properties.deleted) del++;\n'
        '    }\n'
        '    alert("Exported " + features.length + " facilities (" + chk + " checked, " + del + " deleted).");\n'
        '  };\n'
        '\n'
        '  // --- Import ---\n'
        '  window.edImport = function(input) {\n'
        '    var file = input.files[0];\n'
        '    if (!file) return;\n'
        '    var reader = new FileReader();\n'
        '    reader.onload = function(e) {\n'
        '      try {\n'
        '        var fc = JSON.parse(e.target.result);\n'
        '        if (!fc.features) { alert("Invalid GeoJSON: no features array."); return; }\n'
        '      } catch(err) { alert("Failed to parse GeoJSON: " + err.message); return; }\n'
        '      // Restore buffer radius from FeatureCollection or first feature\n'
        '      if (fc.buffer_radius_miles) {\n'
        '        bufferRadiusMiles = fc.buffer_radius_miles;\n'
        '        document.getElementById("ed-buffer-radius").value = bufferRadiusMiles;\n'
        '      } else {\n'
        '        for (var i = 0; i < fc.features.length; i++) {\n'
        '          var bp = fc.features[i].properties || {};\n'
        '          if (bp.buffer_radius_miles && !bp._is_buffer) {\n'
        '            bufferRadiusMiles = bp.buffer_radius_miles;\n'
        '            document.getElementById("ed-buffer-radius").value = bufferRadiusMiles;\n'
        '            break;\n'
        '          }\n'
        '        }\n'
        '      }\n'
        '      // Filter out buffer features\n'
        '      fc.features = fc.features.filter(function(ft) {\n'
        '        return !(ft.properties && ft.properties._is_buffer);\n'
        '      });\n'
        '      // Detect format: new format has checked/deleted properties\n'
        '      var isNewFormat = false;\n'
        '      for (var i = 0; i < fc.features.length; i++) {\n'
        '        var pp = fc.features[i].properties || {};\n'
        '        if (pp.hasOwnProperty("checked") || pp.hasOwnProperty("deleted")) {\n'
        '          isNewFormat = true; break;\n'
        '        }\n'
        '      }\n'
        '      // Build lookup indexes for matching\n'
        '      var byOsm = {}, byFsq = {}, byName = {};\n'
        '      for (var i = 0; i < facilities.length; i++) {\n'
        '        var f = facilities[i];\n'
        '        if (f.osm_id) byOsm[f.osm_id] = f;\n'
        '        if (f.fsq_id) byFsq[f.fsq_id] = f;\n'
        '        byName[f.name] = f;\n'
        '      }\n'
        '      var matched = 0, created = 0;\n'
        '      var m = getMap();\n'
        '      // Reset: uncheck everything, re-show deleted layers, clear deletions\n'
        '      for (var i = 0; i < facilities.length; i++) {\n'
        '        var fac = facilities[i];\n'
        '        fState[fac.id].checked = false;\n'
        '        if (m && layers[fac.id] && !m.hasLayer(layers[fac.id])) {\n'
        '          layers[fac.id].addTo(m);\n'
        '        }\n'
        '      }\n'
        '      deletedIds = {};\n'
        '      // Old format backward compat: process _deleted array\n'
        '      if (!isNewFormat && fc._deleted && fc._deleted.length) {\n'
        '        for (var d = 0; d < fc._deleted.length; d++) {\n'
        '          var dd = fc._deleted[d];\n'
        '          var dt = null;\n'
        '          if (dd.osm_id && byOsm[dd.osm_id]) dt = byOsm[dd.osm_id];\n'
        '          else if (dd.fsq_id && byFsq[dd.fsq_id]) dt = byFsq[dd.fsq_id];\n'
        '          else if (dd.name && byName[dd.name]) dt = byName[dd.name];\n'
        '          if (dt) {\n'
        '            deletedIds[dt.id] = true;\n'
        '            fState[dt.id].checked = false;\n'
        '            if (m && layers[dt.id] && m.hasLayer(layers[dt.id])) m.removeLayer(layers[dt.id]);\n'
        '          }\n'
        '        }\n'
        '      }\n'
        '      // Process features\n'
        '      for (var j = 0; j < fc.features.length; j++) {\n'
        '        var feat = fc.features[j];\n'
        '        var p = feat.properties || {};\n'
        '        var target = null;\n'
        '        if (p.osm_id && byOsm[p.osm_id]) target = byOsm[p.osm_id];\n'
        '        else if (p.fsq_id && byFsq[p.fsq_id]) target = byFsq[p.fsq_id];\n'
        '        else if (p.name && byName[p.name]) target = byName[p.name];\n'
        '        if (target) {\n'
        '          // Matched existing facility\n'
        '          matched++;\n'
        '          fState[target.id].checked = isNewFormat ? !!p.checked : true;\n'
        '          if (isNewFormat && p.deleted) {\n'
        '            deletedIds[target.id] = true;\n'
        '            fState[target.id].checked = false;\n'
        '            if (m && layers[target.id] && m.hasLayer(layers[target.id])) m.removeLayer(layers[target.id]);\n'
        '          }\n'
        '          // Replace geometry with imported version\n'
        '          if (feat.geometry) {\n'
        '            var oldLayer = layers[target.id];\n'
        '            if (oldLayer && m) {\n'
        '              var wasVisible = m.hasLayer(oldLayer);\n'
        '              if (wasVisible) m.removeLayer(oldLayer);\n'
        '              var color = (target.category === "verified") ? "#2563eb" : "#dc2626";\n'
        '              var newGJ = {type:"Feature", geometry: feat.geometry, properties:{id:target.id, name:target.name}};\n'
        '              var newLayer = L.geoJSON(newGJ, {\n'
        '                style: {fillColor: color, color: color, weight: 2, fillOpacity: 0.3}\n'
        '              });\n'
        '              newLayer.eachLayer(function(sub) {\n'
        '                sub.bindTooltip(target.name);\n'
        '                var tid = target.id;\n'
        '                sub.on("contextmenu", function(ev) {\n'
        '                  L.DomEvent.preventDefault(ev);\n'
        '                  showCtxMenu(ev.originalEvent.clientX, ev.originalEvent.clientY, tid);\n'
        '                });\n'
        '                sub.on("mouseover", function() { hoveredId = tid; });\n'
        '                sub.on("mouseout", function() { if (hoveredId === tid) hoveredId = null; });\n'
        '              });\n'
        '              layers[target.id] = newLayer;\n'
        '              if (wasVisible && !deletedIds[target.id]) newLayer.addTo(m);\n'
        '            }\n'
        '          }\n'
        '          if (p.poi_lat != null) target.poi_lat = p.poi_lat;\n'
        '          if (p.poi_lon != null) target.poi_lon = p.poi_lon;\n'
        '          if (p.fsq_name) target.fsq_name = p.fsq_name;\n'
        '          if (p.fsq_address) target.fsq_address = p.fsq_address;\n'
        '          if (p._node_ids !== undefined) target._node_ids = p._node_ids;\n'
        '        } else {\n'
        '          // Unmatched: create new facility\n'
        '          if (!feat.geometry) continue;\n'
        '          created++;\n'
        '          var newId = facilities.length > 0\n'
        '            ? Math.max.apply(null, facilities.map(function(ff){return ff.id;})) + 1 : 0;\n'
        '          var newF = {\n'
        '            id: newId, name: p.name || ("Imported Course " + created),\n'
        '            category: p.category || "fsq_only",\n'
        '            checked: isNewFormat ? !!p.checked : true,\n'
        '            geometry: feat.geometry,\n'
        '            osm_id: p.osm_id || "", fsq_id: p.fsq_id || "",\n'
        '            fsq_name: p.fsq_name || "",\n'
        '            fsq_address: p.fsq_address || "",\n'
        '            poi_lat: (p.poi_lat != null) ? p.poi_lat : null,\n'
        '            poi_lon: (p.poi_lon != null) ? p.poi_lon : null,\n'
        '            _node_ids: p._node_ids || null\n'
        '          };\n'
        '          facilities.push(newF);\n'
        '          fState[newId] = {checked: newF.checked};\n'
        '          if (newF.osm_id) byOsm[newF.osm_id] = newF;\n'
        '          if (newF.fsq_id) byFsq[newF.fsq_id] = newF;\n'
        '          byName[newF.name] = newF;\n'
        '          var color = (newF.category === "verified") ? "#2563eb" : "#dc2626";\n'
        '          var nf = {type:"Feature", geometry: feat.geometry, properties:{id:newId, name:newF.name}};\n'
        '          var nl = L.geoJSON(nf, {\n'
        '            style: {fillColor: color, color: color, weight: 2, fillOpacity: 0.3}\n'
        '          });\n'
        '          (function(cid, cname) {\n'
        '            nl.eachLayer(function(sub) {\n'
        '              sub.bindTooltip(cname);\n'
        '              sub.on("contextmenu", function(ev) {\n'
        '                L.DomEvent.preventDefault(ev);\n'
        '                showCtxMenu(ev.originalEvent.clientX, ev.originalEvent.clientY, cid);\n'
        '              });\n'
        '              sub.on("mouseover", function() { hoveredId = cid; });\n'
        '              sub.on("mouseout", function() { if (hoveredId === cid) hoveredId = null; });\n'
        '            });\n'
        '          })(newId, newF.name);\n'
        '          layers[newId] = nl;\n'
        '          if (isNewFormat && p.deleted) {\n'
        '            deletedIds[newId] = true;\n'
        '            fState[newId].checked = false;\n'
        '          } else if (m) {\n'
        '            nl.addTo(m);\n'
        '          }\n'
        '        }\n'
        '      }\n'
        '      edRenderList();\n'
        '      updateAllBuffers();\n'
        '      var msg = "Imported " + matched + " matched facilities.";\n'
        '      if (created > 0) msg += " Created " + created + " new.";\n'
        '      alert(msg);\n'
        '    };\n'
        '    reader.readAsText(file);\n'
        '    input.value = "";\n'
        '  };\n'
        '\n'
        '  // --- Geodesic area helper (Shoelace on spherical excess) ---\n'
        '  function geodesicArea(coords) {\n'
        '    // coords: array of [lon, lat] pairs (closed ring)\n'
        '    var R = 6371000; // Earth radius in meters\n'
        '    var n = coords.length;\n'
        '    if (n < 4) return 0;\n'
        '    var total = 0;\n'
        '    for (var i = 0; i < n - 1; i++) {\n'
        '      var j = (i + 1) % (n - 1);\n'
        '      var lat1 = coords[i][1] * Math.PI / 180;\n'
        '      var lat2 = coords[j][1] * Math.PI / 180;\n'
        '      var dlon = (coords[j][0] - coords[i][0]) * Math.PI / 180;\n'
        '      total += dlon * (2 + Math.sin(lat1) + Math.sin(lat2));\n'
        '    }\n'
        '    return Math.abs(total * R * R / 2);\n'
        '  }\n'
        '\n'
        '  function computeCentroid(coords) {\n'
        '    // coords: array of [lon, lat] (closed ring) — simple average\n'
        '    var n = coords.length - 1; // exclude closing vertex\n'
        '    if (n <= 0) return [0, 0];\n'
        '    var sumLon = 0, sumLat = 0;\n'
        '    for (var i = 0; i < n; i++) { sumLon += coords[i][0]; sumLat += coords[i][1]; }\n'
        '    return [sumLon / n, sumLat / n];\n'
        '  }\n'
        '\n'
        '  // --- CSV Export ---\n'
        '  function csvEscape(val) {\n'
        '    var s = String(val == null ? "" : val);\n'
        '    if (s.indexOf(",") >= 0 || s.indexOf(\'"\') >= 0 || s.indexOf("\\n") >= 0) {\n'
        '      return \'"\' + s.replace(/"/g, \'""\') + \'"\';\n'
        '    }\n'
        '    return s;\n'
        '  }\n'
        '  window.edExportCSV = function() {\n'
        '    var header = ["name","category","area_sq_meters","area_acres","poi_latitude","poi_longitude","centroid_latitude","centroid_longitude","osm_id","foursquare_verified","foursquare_name","foursquare_fsq_place_id","foursquare_address"];\n'
        '    var rows = [header.join(",")];\n'
        '    for (var i = 0; i < facilities.length; i++) {\n'
        '      var f = facilities[i];\n'
        '      if (!fState[f.id].checked || deletedIds[f.id]) continue;\n'
        '      var layer = layers[f.id];\n'
        '      if (!layer) continue;\n'
        '      var geom = null;\n'
        '      layer.eachLayer(function(sub) { if (sub.toGeoJSON) geom = sub.toGeoJSON().geometry; });\n'
        '      if (!geom) continue;\n'
        '      // Compute area and centroid from geometry\n'
        '      var areaSqM = 0, centLon = 0, centLat = 0;\n'
        '      if (geom.type === "Polygon") {\n'
        '        areaSqM = geodesicArea(geom.coordinates[0]);\n'
        '        var c = computeCentroid(geom.coordinates[0]);\n'
        '        centLon = c[0]; centLat = c[1];\n'
        '      } else if (geom.type === "MultiPolygon") {\n'
        '        var totalArea = 0;\n'
        '        var wLon = 0, wLat = 0;\n'
        '        for (var p = 0; p < geom.coordinates.length; p++) {\n'
        '          var pArea = geodesicArea(geom.coordinates[p][0]);\n'
        '          totalArea += pArea;\n'
        '          var pc = computeCentroid(geom.coordinates[p][0]);\n'
        '          wLon += pc[0] * pArea; wLat += pc[1] * pArea;\n'
        '        }\n'
        '        areaSqM = totalArea;\n'
        '        if (totalArea > 0) { centLon = wLon / totalArea; centLat = wLat / totalArea; }\n'
        '      }\n'
        '      var areaAcres = areaSqM * 0.000247105;\n'
        '      var poiLat = (f.poi_lat != null) ? f.poi_lat : "";\n'
        '      var poiLon = (f.poi_lon != null) ? f.poi_lon : "";\n'
        '      var fsqVerified = (f.category === "verified");\n'
        '      var row = [\n'
        '        csvEscape(f.name), csvEscape(f.category),\n'
        '        areaSqM.toFixed(2), areaAcres.toFixed(2),\n'
        '        poiLat ? Number(poiLat).toFixed(6) : "",\n'
        '        poiLon ? Number(poiLon).toFixed(6) : "",\n'
        '        centLat.toFixed(6), centLon.toFixed(6),\n'
        '        csvEscape(f.osm_id), fsqVerified,\n'
        '        csvEscape(f.fsq_name || ""), csvEscape(f.fsq_id || ""), csvEscape(f.fsq_address || "")\n'
        '      ];\n'
        '      rows.push(row.join(","));\n'
        '    }\n'
        '    if (rows.length <= 1) { alert("No facilities checked for export."); return; }\n'
        '    var blob = new Blob([rows.join("\\n")], {type: "text/csv"});\n'
        '    var url = URL.createObjectURL(blob);\n'
        '    var a = document.createElement("a");\n'
        '    a.href = url; a.download = "EDITED_CSV_FILENAME";\n'
        '    document.body.appendChild(a); a.click();\n'
        '    document.body.removeChild(a); URL.revokeObjectURL(url);\n'
        '    alert("Exported " + (rows.length - 1) + " facilities to CSV.");\n'
        '  };\n'
        '\n'
        '  // --- Add Course modal ---\n'
        '  function makeHexagonJS(lat, lon, radiusM) {\n'
        '    radiusM = radiusM || 75;\n'
        '    var latOff = radiusM / 111000;\n'
        '    var lonOff = radiusM / (111000 * Math.cos(lat * Math.PI / 180));\n'
        '    var coords = [];\n'
        '    for (var i = 0; i < 6; i++) {\n'
        '      var angle = (60 * i - 30) * Math.PI / 180;\n'
        '      coords.push([lon + lonOff * Math.cos(angle), lat + latOff * Math.sin(angle)]);\n'
        '    }\n'
        '    coords.push(coords[0].slice());\n'
        '    return {type: "Polygon", coordinates: [coords]};\n'
        '  }\n'
        '  window.edAddOpen = function() {\n'
        '    document.getElementById("ed-add-name").value = "";\n'
        '    document.getElementById("ed-add-lat").value = "";\n'
        '    document.getElementById("ed-add-lon").value = "";\n'
        '    document.getElementById("ed-add-dup-warn").style.display = "none";\n'
        '    document.getElementById("ed-add-modal").classList.add("ed-show");\n'
        '  };\n'
        '  window.edAddClose = function() {\n'
        '    document.getElementById("ed-add-modal").classList.remove("ed-show");\n'
        '  };\n'
        '  window.edAddCheckDup = function() {\n'
        '    var name = document.getElementById("ed-add-name").value.trim().toLowerCase();\n'
        '    var warn = document.getElementById("ed-add-dup-warn");\n'
        '    if (!name) { warn.style.display = "none"; return; }\n'
        '    for (var i = 0; i < facilities.length; i++) {\n'
        '      if (deletedIds[facilities[i].id]) continue;\n'
        '      if (facilities[i].name.toLowerCase() === name) {\n'
        '        warn.textContent = "A course with this name already exists: " + facilities[i].name;\n'
        '        warn.style.display = "block";\n'
        '        return;\n'
        '      }\n'
        '    }\n'
        '    warn.style.display = "none";\n'
        '  };\n'
        '  window.edAddSubmit = function() {\n'
        '    var name = document.getElementById("ed-add-name").value.trim();\n'
        '    var latStr = document.getElementById("ed-add-lat").value.trim();\n'
        '    var lonStr = document.getElementById("ed-add-lon").value.trim();\n'
        '    if (!name) { alert("Name is required."); return; }\n'
        '    if (!latStr || !lonStr) { alert("Latitude and Longitude are required."); return; }\n'
        '    var lat = parseFloat(latStr), lon = parseFloat(lonStr);\n'
        '    if (isNaN(lat) || isNaN(lon)) { alert("Invalid coordinates."); return; }\n'
        '    // Bounds check\n'
        '    if (lat < BOUNDS_MIN_LAT || lat > BOUNDS_MAX_LAT || lon < BOUNDS_MIN_LON || lon > BOUNDS_MAX_LON) {\n'
        '      if (!confirm("These coordinates appear to be outside the expected area (lat BOUNDS_MIN_LAT\\u2013BOUNDS_MAX_LAT, lon BOUNDS_MIN_LON\\u2013BOUNDS_MAX_LON). Continue anyway?")) return;\n'
        '    }\n'
        '    var hexGeom = makeHexagonJS(lat, lon);\n'
        '    var newId = facilities.length > 0 ? Math.max.apply(null, facilities.map(function(f){return f.id;})) + 1 : 0;\n'
        '    var newF = {\n'
        '      id: newId, name: name, category: "fsq_only", checked: true,\n'
        '      geometry: hexGeom, osm_id: "", fsq_id: "", fsq_name: "", fsq_address: "", poi_lat: lat, poi_lon: lon, _node_ids: null\n'
        '    };\n'
        '    facilities.push(newF);\n'
        '    fState[newId] = {checked: true};\n'
        '    // Create Leaflet layer\n'
        '    var m = getMap();\n'
        '    var color = "#dc2626";\n'
        '    var feat = {type:"Feature", geometry: hexGeom, properties:{id:newId, name:name}};\n'
        '    var layer = L.geoJSON(feat, {\n'
        '      style: {fillColor: color, color: color, weight: 2, fillOpacity: 0.3}\n'
        '    });\n'
        '    layer.eachLayer(function(sub) {\n'
        '      sub.bindTooltip(name);\n'
        '      sub.on("contextmenu", function(ev) {\n'
        '        L.DomEvent.preventDefault(ev);\n'
        '        showCtxMenu(ev.originalEvent.clientX, ev.originalEvent.clientY, newId);\n'
        '      });\n'
        '      sub.on("mouseover", function() { hoveredId = newId; });\n'
        '      sub.on("mouseout", function() { if (hoveredId === newId) hoveredId = null; });\n'
        '    });\n'
        '    layers[newId] = layer;\n'
        '    if (m) {\n'
        '      layer.addTo(m);\n'
        '      var bounds = layer.getBounds();\n'
        '      if (bounds.isValid()) m.fitBounds(bounds, {maxZoom: 16, padding: [40, 40]});\n'
        '    }\n'
        '    edAddClose();\n'
        '    edRenderList();\n'
        '    updateBuffer(newId);\n'
        '    alert("Added " + name + ". Right-click the hexagon to edit its shape.");\n'
        '  };\n'
        '\n'
        '  // --- Help modal ---\n'
        '  window.edHelpOpen = function() {\n'
        '    document.getElementById("ed-help-modal").classList.add("ed-show");\n'
        '  };\n'
        '  window.edHelpClose = function() {\n'
        '    document.getElementById("ed-help-modal").classList.remove("ed-show");\n'
        '  };\n'
        '\n'
        '  // --- Keyboard shortcut ---\n'
        '  document.addEventListener("keydown", function(e) {\n'
        '    if (e.key === "Escape") {\n'
        '      if (document.getElementById("ed-help-modal").classList.contains("ed-show")) { edHelpClose(); return; }\n'
        '      if (document.getElementById("ed-add-modal").classList.contains("ed-show")) { edAddClose(); return; }\n'
        '      if (editingId !== null) stopEditingInternal();\n'
        '    }\n'
        '  });\n'
        '\n'
        '  // --- Delete key on hover ---\n'
        '  document.addEventListener("keydown", function(e) {\n'
        '    if (e.key === "Delete" && hoveredId !== null && !deletedIds[hoveredId] && editingId === null) {\n'
        '      deleteFeature(hoveredId);\n'
        '      hoveredId = null;\n'
        '    }\n'
        '  });\n'
        '\n'
        '  // --- Init ---\n'
        '  setTimeout(function() {\n'
        '    initLayers();\n'
        '    edRenderList();\n'
        '    updateAllBuffers();\n'
        '  }, 800);\n'
        '})();\n'
        '</script>\n'
        '{% endmacro %}'
    ).replace("FACILITIES_JSON", facilities_json
    ).replace("GUIDE_HTML_PLACEHOLDER", guide_html
    ).replace("BOUNDS_MIN_LAT", str(bounds_min_lat)
    ).replace("BOUNDS_MAX_LAT", str(bounds_max_lat)
    ).replace("BOUNDS_MIN_LON", str(bounds_min_lon)
    ).replace("BOUNDS_MAX_LON", str(bounds_max_lon)
    ).replace("EDITED_GEOJSON_FILENAME", config.file_path("edited.geojson")
    ).replace("EDITED_CSV_FILENAME", config.file_path("edited.csv"))

    sidebar = MacroElement()
    sidebar._template = Template(editable_tpl)
    m.get_root().add_child(sidebar)

    map_path = os.path.join(DATA_DIR, config.file_path("editable.html"))
    m.save(map_path)
    print(f"  Editable map saved to {map_path}")
    return m


def create_tables(gdf, fsq_validation, config=None, fsq_df=None):
    """Generate summary CSV table with POI and centroid coordinates."""
    if config is None:
        config = PipelineConfig()
    print("[5/5] Generating summary tables...")

    gdf_projected = gdf.to_crs(config.utm_crs)
    gdf_centroids = gdf_projected.geometry.centroid.to_crs("EPSG:4326")

    records = []

    # OSM-based courses (Verified + OSM Only)
    for idx, row in gdf.iterrows():
        geom_projected = gdf_projected.geometry.iloc[gdf.index.get_loc(idx)]
        area_sqm = geom_projected.area
        area_acres = area_sqm * 0.000247105
        centroid_4326 = gdf_centroids.iloc[gdf.index.get_loc(idx)]

        name = row.get("name", "Unknown")
        osm_id = row.get("osm_id", "")

        # Foursquare match info + POI lat/lon
        fsq_verified = False
        fsq_name = ""
        fsq_place_id = ""
        poi_lat = None
        poi_lon = None
        if not fsq_validation.empty:
            match = fsq_validation[
                (fsq_validation["osm_id"] == osm_id)
                & (fsq_validation["foursquare_match"] == True)
            ]
            if not match.empty:
                fsq_verified = True
                fsq_name = match.iloc[0]["foursquare_name"]
                fsq_place_id = match.iloc[0]["foursquare_fsq_place_id"]
                # Look up the POI lat/lon from the Foursquare source data
                if fsq_df is not None and not fsq_df.empty and fsq_place_id:
                    fsq_match = fsq_df[fsq_df["fsq_place_id"] == fsq_place_id]
                    if not fsq_match.empty:
                        poi_lat = fsq_match.iloc[0]["latitude"]
                        poi_lon = fsq_match.iloc[0]["longitude"]

        category = "verified" if fsq_verified else "osm_only"

        records.append(
            {
                "name": name,
                "category": category,
                "area_sq_meters": round(area_sqm, 2),
                "area_acres": round(area_acres, 2),
                "poi_latitude": round(poi_lat, 6) if pd.notna(poi_lat) else "",
                "poi_longitude": round(poi_lon, 6) if pd.notna(poi_lon) else "",
                "centroid_latitude": round(centroid_4326.y, 6),
                "centroid_longitude": round(centroid_4326.x, 6),
                "osm_id": osm_id,
                "foursquare_verified": fsq_verified,
                "foursquare_name": fsq_name,
                "foursquare_fsq_place_id": fsq_place_id,
            }
        )

    # Foursquare-only entries
    if not fsq_validation.empty:
        fsq_only = fsq_validation[fsq_validation["match_method"] == "foursquare_only"]
        for _, row in fsq_only.iterrows():
            lat = row.get("foursquare_latitude")
            lon = row.get("foursquare_longitude")
            if not (pd.notna(lat) and pd.notna(lon)):
                continue
            records.append(
                {
                    "name": row.get("foursquare_name", "Unknown"),
                    "category": "fsq_only",
                    "area_sq_meters": "",
                    "area_acres": "",
                    "poi_latitude": round(lat, 6),
                    "poi_longitude": round(lon, 6),
                    "centroid_latitude": "",
                    "centroid_longitude": "",
                    "osm_id": "",
                    "foursquare_verified": False,
                    "foursquare_name": row.get("foursquare_name", ""),
                    "foursquare_fsq_place_id": row.get("foursquare_fsq_place_id", ""),
                }
            )

    summary_df = pd.DataFrame(records)
    csv_path = os.path.join(DATA_DIR, config.file_path("table.csv"))
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved summary table ({len(summary_df)} courses) to {csv_path}")
    return summary_df


def export_to_josm(geojson_path, original_geojson_path=None, config=None):
    """Convert an edited GeoJSON file to .osm XML for import into JOSM.

    Reads the edited GeoJSON (exported from the editable map), compares it
    against the original to detect modifications, and produces a .osm file
    containing only changed/new features suitable for JOSM review and upload.

    Args:
        geojson_path: Path to the edited GeoJSON file.
        original_geojson_path: Path to the original GeoJSON for change detection.
            Defaults to data/<file_prefix>_data.geojson.
        config: PipelineConfig instance (defaults to NJ golf courses).
    """
    import xml.etree.ElementTree as ET

    if config is None:
        config = PipelineConfig()
    if original_geojson_path is None:
        original_geojson_path = os.path.join(DATA_DIR, config.file_path("data.geojson"))

    # --- 1. Load data ---
    print(f"[JOSM Export] Loading edited GeoJSON: {geojson_path}")
    with open(geojson_path) as f:
        edited = json.load(f)

    original_by_osm_id = {}
    if os.path.exists(original_geojson_path):
        print(f"[JOSM Export] Loading original GeoJSON: {original_geojson_path}")
        with open(original_geojson_path) as f:
            original = json.load(f)
        for feat in original.get("features", []):
            osm_id = feat.get("properties", {}).get("osm_id", "")
            if osm_id:
                original_by_osm_id[osm_id] = feat
    else:
        print(f"[JOSM Export] Original GeoJSON not found at {original_geojson_path}")
        print("  All features with osm_id will be treated as modified.")

    # --- 2. Detect modifications ---
    def _coords_equal(coords_a, coords_b):
        """Compare two nested coordinate structures with epsilon tolerance."""
        if type(coords_a) != type(coords_b):
            return False
        if isinstance(coords_a, (int, float)):
            return round(coords_a, 7) == round(coords_b, 7)
        if isinstance(coords_a, list):
            if len(coords_a) != len(coords_b):
                return False
            return all(_coords_equal(a, b) for a, b in zip(coords_a, coords_b))
        return coords_a == coords_b

    def _geometry_unchanged(edited_feat, orig_feat):
        """Check if a feature's geometry is unchanged from the original."""
        eg = edited_feat.get("geometry", {})
        og = orig_feat.get("geometry", {})
        if eg.get("type") != og.get("type"):
            return False
        return _coords_equal(eg.get("coordinates"), og.get("coordinates"))

    # Internal property keys to exclude from OSM tags
    INTERNAL_KEYS = {
        "osm_id", "osm_type", "_node_ids", "category", "fsq_id",
        "id", "checked", "name",
    }

    # --- 3. Build .osm XML ---
    osm_root = ET.Element("osm", version="0.6", generator="nj-golf-courses")

    next_negative_id = -1
    stats = {"modified_ways": 0, "modified_relations": 0, "new_ways": 0, "skipped": 0}

    # Track emitted node IDs to avoid duplicates
    emitted_nodes = set()

    for feat in edited.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        osm_id = props.get("osm_id", "")
        node_ids_prop = props.get("_node_ids", None)
        name = props.get("name", "")

        # --- New polygon (no osm_id) ---
        if not osm_id:
            # Only process Polygon or MultiPolygon with actual coordinates
            if geom.get("type") not in ("Polygon", "MultiPolygon"):
                stats["skipped"] += 1
                continue

            if geom["type"] == "Polygon":
                rings = geom["coordinates"]
            else:
                # MultiPolygon: flatten to just the outer rings
                rings = [poly[0] for poly in geom["coordinates"]]

            for ring in rings:
                node_refs = []
                for lon, lat in ring:
                    node_id = next_negative_id
                    next_negative_id -= 1
                    ET.SubElement(osm_root, "node", id=str(node_id),
                                  lat=f"{lat:.7f}", lon=f"{lon:.7f}")
                    node_refs.append(node_id)

                way_id = next_negative_id
                next_negative_id -= 1
                way_el = ET.SubElement(osm_root, "way", id=str(way_id))
                for ref in node_refs:
                    ET.SubElement(way_el, "nd", ref=str(ref))
                ET.SubElement(way_el, "tag", k=config.osm_tag, v=config.osm_value)
                if name:
                    ET.SubElement(way_el, "tag", k="name", v=name)
                ET.SubElement(way_el, "tag", k="source",
                              v="Foursquare;aerial imagery")
                stats["new_ways"] += 1
            continue

        # --- Existing feature: check if modified ---
        orig_feat = original_by_osm_id.get(osm_id)
        if orig_feat and _geometry_unchanged(feat, orig_feat):
            stats["skipped"] += 1
            continue

        # --- Modified relation ---
        if osm_id.startswith("relation/"):
            rel_num = osm_id.split("/")[1]
            print(f"  WARNING: Relation {osm_id} modified — please review "
                  f"manually in JOSM")
            stats["modified_relations"] += 1

            # Emit modified node coordinates if we have _node_ids
            if node_ids_prop and geom.get("type") in ("Polygon", "MultiPolygon"):
                if geom["type"] == "Polygon":
                    coord_rings = geom["coordinates"]
                    nid_rings = node_ids_prop  # [[nid, ...], ...]
                else:
                    # MultiPolygon: node_ids_prop is [[[nid, ...]], [[nid, ...]]]
                    coord_rings = [poly[0] for poly in geom["coordinates"]]
                    nid_rings = [group[0] if isinstance(group, list) and group
                                 and isinstance(group[0], list) else group
                                 for group in node_ids_prop]

                for ring_coords, ring_nids in zip(coord_rings, nid_rings):
                    if not ring_nids:
                        continue
                    for (lon, lat), nid in zip(ring_coords, ring_nids):
                        if nid in emitted_nodes:
                            continue
                        emitted_nodes.add(nid)
                        ET.SubElement(osm_root, "node", id=str(nid),
                                      action="modify",
                                      lat=f"{lat:.7f}", lon=f"{lon:.7f}")
            continue

        # --- Modified way ---
        if osm_id.startswith("way/"):
            way_num = int(osm_id.split("/")[1])
            stats["modified_ways"] += 1

            if geom.get("type") != "Polygon":
                print(f"  WARNING: {osm_id} has unexpected geometry type "
                      f"{geom.get('type')}, skipping")
                stats["modified_ways"] -= 1
                stats["skipped"] += 1
                continue

            coord_ring = geom["coordinates"][0]

            if node_ids_prop and isinstance(node_ids_prop, list) and node_ids_prop:
                nid_ring = node_ids_prop[0]
            else:
                nid_ring = None

            if nid_ring and len(nid_ring) == len(coord_ring):
                # Emit nodes with real IDs
                node_refs = []
                for (lon, lat), nid in zip(coord_ring, nid_ring):
                    if nid not in emitted_nodes:
                        emitted_nodes.add(nid)
                        ET.SubElement(osm_root, "node", id=str(nid),
                                      action="modify",
                                      lat=f"{lat:.7f}", lon=f"{lon:.7f}")
                    node_refs.append(nid)
            else:
                # No node IDs — create new nodes (shouldn't normally happen
                # for existing ways but handle gracefully)
                node_refs = []
                for lon, lat in coord_ring:
                    nid = next_negative_id
                    next_negative_id -= 1
                    ET.SubElement(osm_root, "node", id=str(nid),
                                  lat=f"{lat:.7f}", lon=f"{lon:.7f}")
                    node_refs.append(nid)

            # Emit way element
            way_el = ET.SubElement(osm_root, "way", id=str(way_num),
                                   action="modify")
            for ref in node_refs:
                ET.SubElement(way_el, "nd", ref=str(ref))

            # Preserve original tags
            orig_props = (original_by_osm_id.get(osm_id, {})
                          .get("properties", {}))
            tags_emitted = set()
            # Emit tags from original feature properties (these are the
            # authoritative OSM tags)
            for k, v in orig_props.items():
                if k in INTERNAL_KEYS or k.startswith("_"):
                    continue
                if v is not None and str(v).strip():
                    ET.SubElement(way_el, "tag", k=k, v=str(v))
                    tags_emitted.add(k)
            # Also emit any tags from edited properties not already covered
            for k, v in props.items():
                if k in INTERNAL_KEYS or k.startswith("_"):
                    continue
                if k in tags_emitted:
                    continue
                if v is not None and str(v).strip():
                    ET.SubElement(way_el, "tag", k=k, v=str(v))
                    tags_emitted.add(k)
            # Ensure name tag is present
            if "name" not in tags_emitted and name:
                ET.SubElement(way_el, "tag", k="name", v=name)
            continue

        # Fallback: unrecognised osm_id format
        stats["skipped"] += 1

    # --- 4. Write output ---
    output_path = os.path.join(DATA_DIR, config.file_path("josm.osm"))
    tree = ET.ElementTree(osm_root)
    ET.indent(tree, space="  ")
    with open(output_path, "wb") as f:
        tree.write(f, encoding="UTF-8", xml_declaration=True)

    print(f"\n[JOSM Export] Wrote {output_path}")
    print(f"  Modified ways:      {stats['modified_ways']}")
    print(f"  Modified relations: {stats['modified_relations']} (flagged for manual review)")
    print(f"  New ways:           {stats['new_ways']}")
    print(f"  Skipped (unchanged):{stats['skipped']}")
    return output_path


def main(config=None):
    """Run the full pipeline."""
    if config is None:
        config = PipelineConfig()

    print("=" * 60)
    print(f"{config.display_name} Mapper & Validator")
    print("=" * 60)
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    errors = []

    # Step 1: Fetch OSM data
    try:
        osm_gdf = fetch_osm_golf_courses(config)
    except Exception as e:
        print(f"  ERROR fetching OSM data: {e}")
        errors.append(f"OSM fetch: {e}")
        return

    print()

    # Step 2: Fetch Foursquare data
    try:
        fsq_df = fetch_foursquare_golf_courses(config)
    except Exception as e:
        print(f"  ERROR fetching Foursquare data: {e}")
        errors.append(f"Foursquare fetch: {e}")
        fsq_df = pd.DataFrame(
            columns=["fsq_place_id", "name", "latitude", "longitude", "address", "locality"]
        )

    print()

    # Step 3: Cross-validate
    try:
        fsq_validation = validate_with_foursquare(osm_gdf, fsq_df, config)
    except Exception as e:
        print(f"  ERROR in Foursquare validation: {e}")
        errors.append(f"Foursquare validation: {e}")
        fsq_validation = pd.DataFrame()

    print()

    # Step 4: Create maps
    try:
        create_map(osm_gdf, fsq_validation, config)
    except Exception as e:
        print(f"  ERROR creating map: {e}")
        errors.append(f"Map creation: {e}")

    print()

    try:
        create_editable_map(osm_gdf, fsq_validation, config, fsq_df=fsq_df)
    except Exception as e:
        print(f"  ERROR creating editable map: {e}")
        errors.append(f"Editable map creation: {e}")

    print()

    # Step 5: Create summary tables
    try:
        summary = create_tables(osm_gdf, fsq_validation, config, fsq_df)
    except Exception as e:
        print(f"  ERROR creating tables: {e}")
        errors.append(f"Table creation: {e}")
        summary = None

    print()

    # Final summary
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total {config.keyword.lower()}s found (OSM): {len(osm_gdf)}")
    print(f"Foursquare {config.keyword.lower()}s found: {len(fsq_df)}")

    if not fsq_validation.empty:
        matched = len(fsq_validation[fsq_validation["foursquare_match"] == True])
        total = len(fsq_validation[fsq_validation["osm_id"] != ""])
        if total > 0:
            print(f"Foursquare match rate: {matched}/{total} ({matched/total*100:.1f}%)")

    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\nNo errors encountered.")

    print(f"\nOutput files in {DATA_DIR}/:")
    for f in sorted(os.listdir(DATA_DIR)):
        fpath = os.path.join(DATA_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size:,} bytes)")


if __name__ == "__main__":
    if "--josm" in sys.argv:
        idx = sys.argv.index("--josm")
        if idx + 1 < len(sys.argv):
            edited_path = sys.argv[idx + 1]
        else:
            edited_path = os.path.join(DATA_DIR, PipelineConfig().file_path("edited.geojson"))
        original_path = None
        if "--original" in sys.argv:
            oi = sys.argv.index("--original")
            if oi + 1 < len(sys.argv):
                original_path = sys.argv[oi + 1]
        export_to_josm(edited_path, original_path)
    else:
        main()
