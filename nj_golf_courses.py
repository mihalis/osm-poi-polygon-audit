"""
NJ Golf Course Mapper & Validator

Fetches golf course data from OpenStreetMap, cross-validates with Foursquare
and web sources, produces interactive maps and summary reports.
"""

import json
import math
import os
import re
import sys
import time
import warnings

import duckdb
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from branca.element import Template, MacroElement
from bs4 import BeautifulSoup
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.validation import make_valid
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MAPBOX_TOKEN = os.environ.get("MAPBOX_ACCESS_TOKEN", "")
UTM_CRS = "EPSG:32618"  # UTM Zone 18N for NJ


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


def fetch_osm_golf_courses():
    """Fetch golf course polygons from OpenStreetMap via Overpass API."""
    print("[1/7] Fetching golf course polygons from OpenStreetMap...")

    overpass_servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    query = """
    [out:json][timeout:300];
    area["name"="New Jersey"]["admin_level"="4"]->.nj;
    (
      way["leisure"="golf_course"](area.nj);
      relation["leisure"="golf_course"](area.nj);
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

        name = way.get("tags", {}).get("name", "Unknown Golf Course")
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
        name = rel.get("tags", {}).get("name", "Unknown Golf Course")
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

    geojson_path = os.path.join(DATA_DIR, "nj_golf_courses.geojson")
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    print(f"  Found {len(gdf)} golf courses from OpenStreetMap.")
    print(f"  Saved to {geojson_path}")
    return gdf


def fetch_foursquare_golf_courses():
    """Fetch golf course POIs from Foursquare Open Source Places via S3/DuckDB."""
    print("[2/7] Fetching golf course data from Foursquare OS Places (S3)...")

    s3_path = "s3://fsq-os-places-us-east-1/release/dt=2025-09-09/places/parquet/**/*.parquet"

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-east-1';")
    con.execute("SET s3_url_style='path';")
    con.execute("SET s3_access_key_id='';")
    con.execute("SET s3_secret_access_key='';")

    print("  Querying Foursquare OS Places parquet files from S3...")
    print("  (This scans ~11GB of data remotely, may take a few minutes)")

    df = con.execute(f"""
        SELECT fsq_place_id, name, latitude, longitude, address, locality,
               region, fsq_category_labels, date_closed
        FROM read_parquet('{s3_path}')
        WHERE country = 'US'
          AND region = 'NJ'
          AND fsq_category_labels IS NOT NULL
          AND lower(array_to_string(fsq_category_labels, '|')) LIKE '%golf course%'
          AND lower(array_to_string(fsq_category_labels, '|')) NOT LIKE '%mini golf%'
          AND lower(array_to_string(fsq_category_labels, '|')) NOT LIKE '%miniature golf%'
          AND lower(name) NOT LIKE '%mini golf%'
          AND lower(name) NOT LIKE '%miniature golf%'
          AND lower(name) NOT LIKE '%minigolf%'
    """).fetchdf()

    con.close()

    # Convert category labels list to a readable string
    df["fsq_category_labels"] = df["fsq_category_labels"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else str(x)
    )

    fsq_path = os.path.join(DATA_DIR, "foursquare_golf_courses.parquet")
    df.to_parquet(fsq_path, index=False)
    print(f"  Found {len(df)} golf courses from Foursquare OS Places.")
    print(f"  Saved to {fsq_path}")
    return df


def validate_with_foursquare(osm_gdf, fsq_df):
    """Cross-validate OSM polygons against Foursquare points."""
    print("[3/7] Cross-validating OSM data with Foursquare...")

    if fsq_df.empty:
        print("  No Foursquare data available. Generating OSM-only report.")
        osm_projected = osm_gdf.to_crs(UTM_CRS)
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
        report_path = os.path.join(DATA_DIR, "foursquare_validation_report.csv")
        result.to_csv(report_path, index=False)
        print(f"  Saved validation report to {report_path}")
        print(f"  Summary: {len(records)} OSM-only, 0 Foursquare-only, 0 matched")
        return result

    osm_projected = osm_gdf.to_crs(UTM_CRS)
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
    fsq_projected = fsq_gdf.to_crs(UTM_CRS)

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
    report_path = os.path.join(DATA_DIR, "foursquare_validation_report.csv")
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


def fetch_nj_landcover():
    """Fetch green/natural land cover polygons for NJ from Overpass API."""
    print("  Fetching NJ green land cover data from OpenStreetMap...")

    overpass_servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    query = """
    [out:json][timeout:300];
    area["name"="New Jersey"]["admin_level"="4"]->.nj;
    (
      way["landuse"~"^(forest|farmland|grass|meadow|recreation_ground|orchard|vineyard)$"](area.nj);
      way["leisure"~"^(park|garden|nature_reserve)$"](area.nj);
      way["natural"~"^(wood|scrub|grassland|heath)$"](area.nj);
    );
    out geom;
    """

    data = None
    for url in overpass_servers:
        try:
            print(f"    Trying {url}...")
            resp = requests.post(url, data={"data": query}, timeout=360)
            resp.raise_for_status()
            data = resp.json()
            print(f"    Success ({len(data.get('elements', []))} elements)")
            break
        except Exception as e:
            print(f"    Failed: {e}")

    if data is None:
        return None

    features = []
    for el in data.get("elements", []):
        if el["type"] != "way" or "geometry" not in el:
            continue
        coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        tags = el.get("tags", {})
        cat = tags.get("landuse") or tags.get("leisure") or tags.get("natural") or "unknown"
        features.append({
            "type": "Feature",
            "properties": {"category": cat},
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })

    if not features:
        return None

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    print(f"  Fetched {len(gdf)} green/natural land cover polygons.")
    return gdf


def validate_foursquare_only(osm_gdf, fsq_validation):
    """Score Foursquare-only POIs by likelihood of being actual golf courses.

    Uses surrounding land cover (green vs urban) to predict whether a
    Foursquare-only point is a real golf course. Trains a model on verified
    OSM+Foursquare golf course centroids as positive examples.
    """
    print("[3b/7] Validating Foursquare-only POIs with land cover analysis...")

    fsq_only_mask = fsq_validation["match_method"] == "foursquare_only"
    n_fsq_only = fsq_only_mask.sum()

    # Initialize golf_probability column
    fsq_validation["golf_probability"] = np.nan
    fsq_validation.loc[fsq_validation["foursquare_match"] == True, "golf_probability"] = 1.0
    fsq_validation.loc[
        (fsq_validation["match_method"] == "none") & (fsq_validation["osm_id"] != ""),
        "golf_probability",
    ] = 1.0  # OSM-only courses are real courses too

    if n_fsq_only == 0:
        print("  No Foursquare-only POIs to validate.")
        return fsq_validation

    # Fetch green land cover for NJ
    landcover_gdf = fetch_nj_landcover()

    if landcover_gdf is None or landcover_gdf.empty:
        print("  Land cover data unavailable. Setting default probability.")
        fsq_validation.loc[fsq_only_mask, "golf_probability"] = 0.5
        return fsq_validation

    # Project land cover and fix invalid geometries
    lc_proj = landcover_gdf.to_crs(UTM_CRS)
    lc_proj["geometry"] = lc_proj.geometry.apply(
        lambda g: make_valid(g) if g and not g.is_empty and not g.is_valid else g
    )
    lc_proj = lc_proj[~lc_proj.geometry.is_empty & lc_proj.geometry.notna()]

    sindex = lc_proj.sindex
    BUFFER_M = 500
    buffer_area = np.pi * BUFFER_M ** 2

    def compute_features(lat, lon):
        """Compute land cover features within 500m buffer around a point."""
        try:
            pt_proj = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(UTM_CRS)[0]
            buf = pt_proj.buffer(BUFFER_M)
            candidates_idx = list(sindex.intersection(buf.bounds))
            if not candidates_idx:
                return [0.0, 0.0, 0]
            candidates = lc_proj.iloc[candidates_idx]
            hits = candidates[candidates.intersects(buf)]
            if hits.empty:
                return [0.0, 0.0, 0]
            clipped = hits.intersection(buf)
            areas = clipped.area
            green_ratio = min(areas.sum() / buffer_area, 1.0)
            max_patch_ratio = min(areas.max() / buffer_area, 1.0)
            patch_count = len(areas)
            return [green_ratio, max_patch_ratio, patch_count]
        except Exception:
            return [0.0, 0.0, 0]

    # Positive examples: centroids of verified OSM golf course polygons
    print("  Computing features for verified golf courses (positive examples)...")
    osm_proj = osm_gdf.to_crs(UTM_CRS)
    osm_proj["geometry"] = osm_proj.geometry.apply(
        lambda g: make_valid(g) if not g.is_valid else g
    )

    verified_osm_ids = set(
        fsq_validation[fsq_validation["foursquare_match"] == True]["osm_id"].values
    )

    pos_features = []
    for idx, row in osm_gdf.iterrows():
        if row.get("osm_id", "") in verified_osm_ids:
            c = row.geometry.centroid
            pos_features.append(compute_features(c.y, c.x))
    print(f"    {len(pos_features)} positive examples computed")

    # Negative examples: random points in NJ far from any golf course
    print("  Generating negative training examples...")
    np.random.seed(42)
    bounds = osm_gdf.total_bounds  # minx, miny, maxx, maxy

    neg_features = []
    target_neg = min(len(pos_features), 200)
    attempts = 0
    while len(neg_features) < target_neg and attempts < 3000:
        rlon = np.random.uniform(bounds[0], bounds[2])
        rlat = np.random.uniform(bounds[1], bounds[3])
        rpt = gpd.GeoSeries([Point(rlon, rlat)], crs="EPSG:4326").to_crs(UTM_CRS)[0]
        min_dist = osm_proj.geometry.distance(rpt).min()
        if min_dist > 2000:  # At least 2km from any golf course
            neg_features.append(compute_features(rlat, rlon))
        attempts += 1
    print(f"    {len(neg_features)} negative examples computed")

    if len(pos_features) < 10 or len(neg_features) < 10:
        print("  Not enough training data. Setting default probability.")
        fsq_validation.loc[fsq_only_mask, "golf_probability"] = 0.5
        return fsq_validation

    # Train model
    print(f"  Training GradientBoosting model...")
    X = np.array(pos_features + neg_features)
    y = np.array([1] * len(pos_features) + [0] * len(neg_features))

    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)
    train_acc = model.score(X, y)
    print(f"    Training accuracy: {train_acc:.1%}")
    print(f"    Feature importances: green_ratio={model.feature_importances_[0]:.2f}, "
          f"max_patch={model.feature_importances_[1]:.2f}, "
          f"patch_count={model.feature_importances_[2]:.2f}")

    # Score FSQ-only POIs
    print(f"  Scoring {n_fsq_only} Foursquare-only POIs...")
    fsq_only_indices = fsq_validation.index[fsq_only_mask]
    probs = []
    for i, idx in enumerate(fsq_only_indices):
        row = fsq_validation.loc[idx]
        lat = row.get("foursquare_latitude")
        lon = row.get("foursquare_longitude")
        if pd.notna(lat) and pd.notna(lon):
            feats = compute_features(lat, lon)
            prob = model.predict_proba(np.array([feats]))[0][1]
        else:
            prob = 0.0
        probs.append(prob)
        if (i + 1) % 100 == 0:
            print(f"    Scored {i + 1}/{n_fsq_only}...")

    fsq_validation.loc[fsq_only_indices, "golf_probability"] = probs

    likely = sum(1 for p in probs if p >= 0.5)
    print(f"  Results: {likely} likely golf courses, {n_fsq_only - likely} unlikely")

    # Save updated report
    report_path = os.path.join(DATA_DIR, "foursquare_validation_report.csv")
    fsq_validation.to_csv(report_path, index=False)
    print(f"  Updated {report_path}")

    return fsq_validation


def _fetch_buildings_around_points(query_points, radius_m):
    """Fetch building center points from Overpass in batches.

    Args:
        query_points: list of (lat, lon) tuples
        radius_m: search radius in meters

    Returns:
        dict mapping osm_id -> (lat, lon) for deduplicated building centers
    """
    overpass_servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    BATCH_SIZE = 10
    all_buildings = {}  # osm_id -> (lat, lon)

    batches = [
        query_points[i : i + BATCH_SIZE]
        for i in range(0, len(query_points), BATCH_SIZE)
    ]

    for batch_idx, batch in enumerate(batches):
        around_parts = "\n".join(
            f'way["building"](around:{radius_m},{lat},{lon});'
            for lat, lon in batch
        )
        query = f"""
        [out:json][timeout:180];
        (
          {around_parts}
        );
        out center qt;
        """

        success = False
        for url in overpass_servers:
            try:
                resp = requests.post(url, data={"data": query}, timeout=240)
                resp.raise_for_status()
                data = resp.json()
                for el in data.get("elements", []):
                    if "center" in el:
                        all_buildings[el["id"]] = (
                            el["center"]["lat"],
                            el["center"]["lon"],
                        )
                success = True
                break
            except Exception as e:
                if url == overpass_servers[-1]:
                    print(f"    Batch {batch_idx + 1}: all servers failed - {e}")
                continue

        time.sleep(1.5)
        if (batch_idx + 1) % 10 == 0:
            print(
                f"    Fetched {batch_idx + 1}/{len(batches)} batches "
                f"({len(all_buildings)} buildings so far)..."
            )

    return all_buildings


def validate_foursquare_only_v2(osm_gdf, fsq_validation, fsq_df):
    """V2: Score FSQ-only POIs using building-free directional ellipse search.

    Approach:
    1. Learn typical golf course geometry from verified OSM+Foursquare matches
       (area, aspect ratio, POI position relative to polygon).
    2. Fetch building footprints around each POI from Overpass.
    3. For each FSQ-only POI, try placing an ellipse of typical golf course
       size in 16 directions. The POI sits at the edge of the ellipse (since
       the clubhouse is typically at the course boundary). Count buildings
       inside each candidate ellipse.
    4. Score based on the minimum building count found (fewer = more likely
       to be a real golf course).
    """
    print("=" * 60)
    print("[V2] Building-based directional ellipse validation")
    print("=" * 60)

    fsq_only_mask = fsq_validation["match_method"] == "foursquare_only"
    n_fsq_only = fsq_only_mask.sum()

    # Initialize v2 probability column
    fsq_validation["golf_probability_v2"] = np.nan
    fsq_validation.loc[
        fsq_validation["foursquare_match"] == True, "golf_probability_v2"
    ] = 1.0
    fsq_validation.loc[
        (fsq_validation["match_method"] == "none")
        & (fsq_validation["osm_id"] != ""),
        "golf_probability_v2",
    ] = 1.0

    if n_fsq_only == 0:
        print("  No Foursquare-only POIs to validate.")
        return fsq_validation

    # === Phase 1: Learn golf course geometry from verified matches ===
    print("\n  Phase 1: Learning golf course geometry from verified matches...")

    osm_proj = osm_gdf.to_crs(UTM_CRS)
    osm_proj["geometry"] = osm_proj.geometry.apply(
        lambda g: make_valid(g) if not g.is_valid else g
    )

    # Get FSQ coordinates for verified courses
    fsq_coords = fsq_df[["fsq_place_id", "latitude", "longitude"]].copy()
    fsq_coords = fsq_coords.dropna(subset=["latitude", "longitude"])

    verified = fsq_validation[fsq_validation["foursquare_match"] == True].copy()

    areas = []
    aspect_ratios = []
    poi_offsets = []  # distance from POI to polygon centroid

    for _, v_row in verified.iterrows():
        osm_id = v_row["osm_id"]
        fsq_id = v_row["foursquare_fsq_place_id"]
        fsq_match = fsq_coords[fsq_coords["fsq_place_id"] == fsq_id]
        if fsq_match.empty:
            continue
        lat, lon = fsq_match.iloc[0]["latitude"], fsq_match.iloc[0]["longitude"]

        osm_rows = osm_gdf[osm_gdf["osm_id"] == osm_id]
        if osm_rows.empty:
            continue
        loc = osm_gdf.index.get_loc(osm_rows.index[0])
        poly = osm_proj.geometry.iloc[loc]
        if poly is None or poly.is_empty:
            continue

        poi = gpd.GeoSeries(
            [Point(lon, lat)], crs="EPSG:4326"
        ).to_crs(UTM_CRS)[0]

        areas.append(poly.area)
        poi_offsets.append(poly.centroid.distance(poi))

        b = poly.bounds
        dx, dy = b[2] - b[0], b[3] - b[1]
        if min(dx, dy) > 0:
            aspect_ratios.append(max(dx, dy) / min(dx, dy))

    if len(areas) < 10:
        print("  Not enough verified course geometry. Skipping v2.")
        fsq_validation.loc[fsq_only_mask, "golf_probability_v2"] = 0.5
        return fsq_validation

    median_area = np.median(areas)
    median_aspect = np.median(aspect_ratios) if aspect_ratios else 1.5
    median_offset = np.median(poi_offsets)

    # Ellipse semi-axes from learned geometry
    semi_major = np.sqrt(median_area * median_aspect / np.pi)
    semi_minor = np.sqrt(median_area / (np.pi * median_aspect))
    # Search radius: ellipse extends up to 2*semi_major from POI + margin
    search_radius = int(np.sqrt((2 * semi_major) ** 2 + semi_minor**2) + 200)

    print(f"    Verified courses analyzed: {len(areas)}")
    print(f"    Median area: {median_area:,.0f} sq m ({median_area * 0.000247105:.0f} acres)")
    print(f"    Median aspect ratio: {median_aspect:.2f}")
    print(f"    Median POI-to-centroid offset: {median_offset:.0f} m")
    print(f"    Search ellipse: {semi_major:.0f}m x {semi_minor:.0f}m (semi-axes)")
    print(f"    Building search radius: {search_radius}m")

    # === Phase 2: Fetch building center points ===
    print("\n  Phase 2: Fetching building footprints from OpenStreetMap...")

    # Collect all coordinates we need buildings around
    all_query_coords = []

    # FSQ-only coordinates
    fsq_only_data = fsq_validation[fsq_only_mask]
    for _, row in fsq_only_data.iterrows():
        lat = row.get("foursquare_latitude")
        lon = row.get("foursquare_longitude")
        if pd.notna(lat) and pd.notna(lon):
            all_query_coords.append((lat, lon))

    # Sample verified course coordinates (for reference scoring)
    verified_query_coords = []
    for _, v_row in verified.iterrows():
        fsq_id = v_row["foursquare_fsq_place_id"]
        match = fsq_coords[fsq_coords["fsq_place_id"] == fsq_id]
        if not match.empty:
            lat, lon = match.iloc[0]["latitude"], match.iloc[0]["longitude"]
            verified_query_coords.append((lat, lon))
    # Use a sample of verified for building queries (limit Overpass load)
    np.random.seed(42)
    if len(verified_query_coords) > 80:
        sample_idx = np.random.choice(
            len(verified_query_coords), 80, replace=False
        )
        verified_sample_coords = [verified_query_coords[i] for i in sample_idx]
    else:
        verified_sample_coords = verified_query_coords
    all_query_coords.extend(verified_sample_coords)

    # Random negative examples (urban points far from golf courses)
    bounds = osm_gdf.total_bounds
    neg_coords = []
    neg_attempts = 0
    while len(neg_coords) < 50 and neg_attempts < 500:
        rlon = np.random.uniform(bounds[0], bounds[2])
        rlat = np.random.uniform(bounds[1], bounds[3])
        rpt = gpd.GeoSeries(
            [Point(rlon, rlat)], crs="EPSG:4326"
        ).to_crs(UTM_CRS)[0]
        if osm_proj.geometry.distance(rpt).min() > 2000:
            neg_coords.append((rlat, rlon))
            all_query_coords.append((rlat, rlon))
        neg_attempts += 1

    # Deduplicate coordinates (within ~10m of each other)
    unique_coords = []
    seen = set()
    for lat, lon in all_query_coords:
        key = (round(lat, 4), round(lon, 4))
        if key not in seen:
            seen.add(key)
            unique_coords.append((lat, lon))

    print(f"    Querying buildings around {len(unique_coords)} unique locations...")
    building_dict = _fetch_buildings_around_points(unique_coords, search_radius)
    print(f"    Total unique building points: {len(building_dict)}")

    if not building_dict:
        print("  No building data fetched. Using default scores.")
        fsq_validation.loc[fsq_only_mask, "golf_probability_v2"] = 0.5
        return fsq_validation

    # Build spatial index on building center points
    bldg_latlons = list(building_dict.values())
    bldg_points_geom = [Point(lon, lat) for lat, lon in bldg_latlons]
    bldg_gdf = gpd.GeoDataFrame(
        geometry=bldg_points_geom, crs="EPSG:4326"
    ).to_crs(UTM_CRS)
    bldg_sindex = bldg_gdf.sindex

    # === Phase 3: Directional ellipse scoring ===
    print("\n  Phase 3: Directional ellipse scoring...")

    N_DIRECTIONS = 16
    angles = [2 * np.pi * i / N_DIRECTIONS for i in range(N_DIRECTIONS)]

    def score_point(lat, lon):
        """Find most building-free ellipse direction from a POI.

        Returns (min_buildings_in_best_ellipse, surrounding_density).
        """
        try:
            poi_proj = gpd.GeoSeries(
                [Point(lon, lat)], crs="EPSG:4326"
            ).to_crs(UTM_CRS)[0]
        except Exception:
            return 999, 1.0

        best_count = float("inf")

        for angle in angles:
            # Ellipse center: POI + offset in direction `angle`
            # The POI sits at the edge of the ellipse
            cx = poi_proj.x + semi_major * np.cos(angle)
            cy = poi_proj.y + semi_major * np.sin(angle)

            # Conservative bounding box for spatial index query
            bbox = (
                cx - semi_major,
                cy - semi_major,
                cx + semi_major,
                cy + semi_major,
            )
            candidates_idx = list(bldg_sindex.intersection(bbox))

            count = 0
            for bidx in candidates_idx:
                bpt = bldg_gdf.geometry.iloc[bidx]
                # Translate relative to ellipse center
                dx = bpt.x - cx
                dy = bpt.y - cy
                # Rotate by -angle to align with ellipse axes
                cos_a = np.cos(-angle)
                sin_a = np.sin(-angle)
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                # Ellipse equation
                if (rx / semi_major) ** 2 + (ry / semi_minor) ** 2 <= 1:
                    count += 1

            best_count = min(best_count, count)

        # Surrounding building density (per sq km)
        sr_bounds = (
            poi_proj.x - search_radius,
            poi_proj.y - search_radius,
            poi_proj.x + search_radius,
            poi_proj.y + search_radius,
        )
        total_nearby = len(list(bldg_sindex.intersection(sr_bounds)))
        density = total_nearby / (np.pi * search_radius**2) * 1e6

        return best_count, density

    # Score verified courses (positive examples)
    print("    Scoring verified sample (positive examples)...")
    pos_scores = []
    for lat, lon in verified_sample_coords:
        mb, dens = score_point(lat, lon)
        pos_scores.append([mb, dens])
    print(f"      {len(pos_scores)} verified courses scored")
    pos_arr = np.array(pos_scores)
    print(
        f"      Min buildings in best ellipse: "
        f"median={np.median(pos_arr[:, 0]):.0f}, "
        f"mean={np.mean(pos_arr[:, 0]):.0f}, "
        f"p75={np.percentile(pos_arr[:, 0], 75):.0f}"
    )

    # Score negative examples
    print("    Scoring negative examples...")
    neg_scores = []
    for lat, lon in neg_coords:
        mb, dens = score_point(lat, lon)
        neg_scores.append([mb, dens])
    print(f"      {len(neg_scores)} negative examples scored")
    if neg_scores:
        neg_arr = np.array(neg_scores)
        print(
            f"      Min buildings in best ellipse: "
            f"median={np.median(neg_arr[:, 0]):.0f}, "
            f"mean={np.mean(neg_arr[:, 0]):.0f}"
        )

    if len(pos_scores) < 10 or len(neg_scores) < 10:
        print("  Insufficient training data. Using heuristic scores.")
        fsq_validation.loc[fsq_only_mask, "golf_probability_v2"] = 0.5
        return fsq_validation

    # === Phase 4: Train model ===
    print("\n  Phase 4: Training model...")
    X = np.array(pos_scores + neg_scores)
    y = np.array([1] * len(pos_scores) + [0] * len(neg_scores))

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42
    )
    model.fit(X, y)
    print(f"    Training accuracy: {model.score(X, y):.1%}")
    print(
        f"    Feature importances: min_buildings={model.feature_importances_[0]:.2f}, "
        f"density={model.feature_importances_[1]:.2f}"
    )

    # Score FSQ-only POIs
    print(f"\n  Scoring {n_fsq_only} Foursquare-only POIs...")
    fsq_only_indices = fsq_validation.index[fsq_only_mask]
    probs_v2 = []
    for i, idx in enumerate(fsq_only_indices):
        row = fsq_validation.loc[idx]
        lat = row.get("foursquare_latitude")
        lon = row.get("foursquare_longitude")
        if pd.notna(lat) and pd.notna(lon):
            mb, dens = score_point(lat, lon)
            prob = model.predict_proba(np.array([[mb, dens]]))[0][1]
        else:
            prob = 0.0
        probs_v2.append(prob)
        if (i + 1) % 100 == 0:
            print(f"    Scored {i + 1}/{n_fsq_only}...")

    fsq_validation.loc[fsq_only_indices, "golf_probability_v2"] = probs_v2

    likely_v2 = sum(1 for p in probs_v2 if p >= 0.5)
    print(f"\n  V2 Results: {likely_v2} likely golf courses, {n_fsq_only - likely_v2} unlikely")

    # Save v2 report (separate file)
    report_path = os.path.join(DATA_DIR, "foursquare_validation_report_v2.csv")
    fsq_validation.to_csv(report_path, index=False)
    print(f"  Saved to {report_path}")

    # === Print v1 vs v2 comparison for key examples ===
    print("\n  V1 vs V2 comparison (Foursquare-only POIs):")
    print(f"  {'Name':<40} {'V1':>6} {'V2':>6} {'Change':>8}")
    print("  " + "-" * 62)
    for _, row in fsq_validation[fsq_only_mask].iterrows():
        name = row.get("foursquare_name", "")
        v1 = row.get("golf_probability", np.nan)
        v2 = row.get("golf_probability_v2", np.nan)
        if pd.notna(v1) and pd.notna(v2):
            diff = v2 - v1
            # Show key examples and interesting changes
            if any(kw in name for kw in [
                "Avalon", "Union League", "Sand Barrens",
                "Cape May Par", "Pinelands",
            ]) or abs(diff) > 0.3:
                print(f"  {name:<40} {v1:>5.0%} {v2:>5.0%} {diff:>+7.0%}")

    return fsq_validation


def validate_with_web(osm_gdf):
    """Validate golf courses using web search/scraping."""
    print("[4/7] Validating golf courses with web sources...")

    results = []
    osm_projected = osm_gdf.to_crs(UTM_CRS)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for idx, row in osm_gdf.iterrows():
        name = row.get("name", "Unknown Golf Course")
        osm_geom = osm_projected.geometry.iloc[osm_gdf.index.get_loc(idx)]
        area_sqm = osm_geom.area
        area_acres = area_sqm * 0.000247105

        result = {
            "name": name,
            "osm_area_sqm": round(area_sqm, 2),
            "osm_area_acres": round(area_acres, 2),
            "web_area_acres": None,
            "area_diff_pct": None,
            "status": "unknown",
            "course_type": "unknown",
            "holes": "unknown",
            "source_url": "",
            "notes": "",
        }

        if name == "Unknown Golf Course":
            results.append(result)
            continue

        # Try searching for the course info
        try:
            search_query = f"{name} New Jersey golf course"
            search_url = f"https://www.google.com/search?q={requests.utils.quote(search_query)}"
            resp = requests.get(search_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                text = soup.get_text().lower()

                # Try to determine course type
                if "public" in text and "course" in text:
                    result["course_type"] = "public"
                elif "private" in text and ("member" in text or "club" in text):
                    result["course_type"] = "private"
                elif "semi-private" in text or "semi private" in text:
                    result["course_type"] = "semi-private"

                # Try to determine holes
                if "18-hole" in text or "18 hole" in text:
                    result["holes"] = "18"
                elif "27-hole" in text or "27 hole" in text:
                    result["holes"] = "27"
                elif "36-hole" in text or "36 hole" in text:
                    result["holes"] = "36"
                elif "9-hole" in text or "9 hole" in text:
                    result["holes"] = "9"

                # Try to determine status
                if "permanently closed" in text or "closed permanently" in text:
                    result["status"] = "permanently closed"
                elif "temporarily closed" in text:
                    result["status"] = "temporarily closed"
                elif "open" in text and ("tee time" in text or "book" in text):
                    result["status"] = "open"
                else:
                    result["status"] = "likely open"

                # Try to find acreage
                acre_patterns = [
                    r"(\d+[\.,]?\d*)\s*(?:acre|acres)",
                    r"(\d+[\.,]?\d*)\s*-\s*acre",
                ]
                for pattern in acre_patterns:
                    match = re.search(pattern, text)
                    if match:
                        try:
                            web_acres = float(
                                match.group(1).replace(",", "")
                            )
                            if 10 < web_acres < 1000:  # Sanity check
                                result["web_area_acres"] = web_acres
                                diff = abs(area_acres - web_acres) / web_acres * 100
                                result["area_diff_pct"] = round(diff, 1)
                                break
                        except ValueError:
                            pass

                result["source_url"] = search_url
        except Exception as e:
            result["notes"] = f"Web search failed: {e}"

        results.append(result)

        # Rate limit
        time.sleep(1.0)

        # Print progress every 10 courses
        if (len(results)) % 10 == 0:
            print(f"  Processed {len(results)}/{len(osm_gdf)} courses...")

    web_df = pd.DataFrame(results)

    # Generate markdown report
    report_lines = ["# NJ Golf Courses - Web Validation Report\n"]
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    report_lines.append(f"Total courses analyzed: {len(web_df)}\n")

    # Summary statistics
    validated = web_df[web_df["web_area_acres"].notna()]
    with_type = web_df[web_df["course_type"] != "unknown"]
    with_status = web_df[web_df["status"] != "unknown"]
    discrepancies = web_df[
        (web_df["area_diff_pct"].notna()) & (web_df["area_diff_pct"] > 20)
    ]

    report_lines.append("## Summary Statistics\n")
    report_lines.append(f"- Courses with web-reported acreage: {len(validated)}")
    report_lines.append(f"- Courses with identified type: {len(with_type)}")
    report_lines.append(f"- Courses with identified status: {len(with_status)}")
    report_lines.append(
        f"- Courses with significant area discrepancy (>20%): {len(discrepancies)}\n"
    )

    # Course type breakdown
    report_lines.append("## Course Type Breakdown\n")
    type_counts = web_df["course_type"].value_counts()
    for ctype, count in type_counts.items():
        report_lines.append(f"- {ctype}: {count}")
    report_lines.append("")

    # Area comparison table
    report_lines.append("## Area Comparison: OSM vs Web-Reported\n")
    report_lines.append(
        "| Course Name | OSM Area (acres) | Web Area (acres) | Difference (%) | Notes |"
    )
    report_lines.append("|---|---|---|---|---|")
    for _, row in web_df.iterrows():
        web_acres = (
            f"{row['web_area_acres']:.1f}" if pd.notna(row["web_area_acres"]) else "N/A"
        )
        diff = (
            f"{row['area_diff_pct']:.1f}%" if pd.notna(row["area_diff_pct"]) else "N/A"
        )
        flag = " **DISCREPANCY**" if pd.notna(row["area_diff_pct"]) and row["area_diff_pct"] > 20 else ""
        report_lines.append(
            f"| {row['name']} | {row['osm_area_acres']:.1f} | {web_acres} | {diff}{flag} | {row['course_type']}, {row['holes']} holes, {row['status']} |"
        )
    report_lines.append("")

    # Discrepancy details
    if len(discrepancies) > 0:
        report_lines.append("## Significant Area Discrepancies (>20%)\n")
        for _, row in discrepancies.iterrows():
            report_lines.append(f"### {row['name']}")
            report_lines.append(f"- OSM area: {row['osm_area_acres']:.1f} acres")
            report_lines.append(f"- Web-reported area: {row['web_area_acres']:.1f} acres")
            report_lines.append(f"- Difference: {row['area_diff_pct']:.1f}%")
            report_lines.append(
                f"- Possible reasons: OSM boundary may include/exclude parking, clubhouse, driving range, or surrounding land"
            )
            report_lines.append("")

    # Sources
    report_lines.append("## Sources\n")
    report_lines.append("- OpenStreetMap (OSM) polygon data via Overpass API")
    report_lines.append("- Google Search results for individual course information")
    report_lines.append(
        "- Note: Web-scraped data may be incomplete or inaccurate. Manual verification recommended for critical use.\n"
    )

    report_path = os.path.join(DATA_DIR, "web_validation_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"  Validated {len(validated)} courses with web acreage data.")
    print(f"  Found {len(discrepancies)} significant discrepancies.")
    print(f"  Saved report to {report_path}")
    return web_df


def create_map(gdf, validation_df, web_df=None):
    """Create an interactive HTML map with folium and a left sidebar."""
    print("[5/7] Creating interactive map...")

    # Center map on NJ
    gdf_projected = gdf.to_crs(UTM_CRS)
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

    # Build web info lookup
    web_info = {}
    if web_df is not None and not web_df.empty:
        for _, row in web_df.iterrows():
            web_info[row["name"]] = row

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

        if name in web_info:
            wi = web_info[name]
            popup_html += f"Type: {wi.get('course_type', 'unknown')}<br>"
            popup_html += f"Holes: {wi.get('holes', 'unknown')}<br>"
            popup_html += f"Status: {wi.get('status', 'unknown')}<br>"
            if pd.notna(wi.get("web_area_acres")):
                popup_html += f"Web Area: {wi['web_area_acres']:.1f} acres<br>"

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
        '    <h3>NJ Golf Courses</h3>\n'
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
        '    <input type="text" id="search-box" placeholder="Search golf course name..." oninput="filterFacilities()">\n'
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

    map_path = os.path.join(DATA_DIR, "nj_golf_courses_map.html")
    m.save(map_path)
    print(f"  Map saved to {map_path}")
    return m


def create_editable_map(gdf, validation_df):
    """Create an editable HTML map for reviewing/exporting golf course polygons.

    Users can check/uncheck courses, edit polygon vertices via Leaflet-Geoman,
    expand Foursquare-only hexagons into full polygons, and export checked
    polygons as a GeoJSON download.
    """
    print("[Editable Map] Creating editable golf course map...")

    # --- 1. Base map setup ---
    gdf_projected = gdf.to_crs(UTM_CRS)
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

        # Look up fsq_id for verified courses
        fsq_id = ""
        if is_verified and not validation_df.empty:
            match = validation_df[
                (validation_df["osm_id"] == osm_id)
                & (validation_df["foursquare_match"] == True)
            ]
            if not match.empty:
                fsq_id = str(match.iloc[0].get("foursquare_fsq_place_id", ""))

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
            hex_geom = _make_hexagon(float(lat), float(lon), radius_m=75)
            facilities.append({
                "id": fid,
                "name": fsq_name,
                "category": "fsq_only",
                "checked": False,
                "geometry": hex_geom,
                "osm_id": "",
                "fsq_id": fsq_place_id,
            })
            fid += 1

    facilities_json = json.dumps(facilities)

    n_verified = sum(1 for f in facilities if f["category"] == "verified")
    n_osm = sum(1 for f in facilities if f["category"] == "osm_only")
    n_fsq = sum(1 for f in facilities if f["category"] == "fsq_only")
    print(f"  Facilities: {n_verified} verified, {n_osm} OSM-only, {n_fsq} FSQ-only")

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
        '<link rel="stylesheet" href="https://unpkg.com/@geoman-io/leaflet-geoman-free@latest/dist/leaflet-geoman.css" />\n'
        '<script src="https://unpkg.com/@geoman-io/leaflet-geoman-free@latest/dist/leaflet-geoman.min.js"></script>\n'
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
        '.ed-cat-header input[type=checkbox] { margin-right: 8px; cursor: pointer; }\n'
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
        '</style>\n'
        '<div id="ed-ctx-menu">\n'
        '  <div onclick="edCtxEdit()">Edit Shape</div>\n'
        '  <div onclick="edCtxSimplify()">Simplify Vertices</div>\n'
        '  <div class="ed-ctx-danger" onclick="edCtxDelete()">Delete</div>\n'
        '</div>\n'
        '<div id="ed-edit-bar">\n'
        '  Editing: <span class="ed-edit-name" id="ed-edit-name"></span>\n'
        '  <span style="color:#92400e;font-size:11px;">(drag vertices to reshape &mdash; right-click or Esc to finish)</span>\n'
        '  <button class="ed-done-btn" onclick="edStopEditing()">Done</button>\n'
        '</div>\n'
        '<div id="ed-sidebar">\n'
        '  <div id="ed-sidebar-header">\n'
        '    <h3>Editable Golf Courses</h3>\n'
        '    <div class="ed-hint">Checkbox = include in export. Right-click polygon to edit or delete.</div>\n'
        '    <div class="ed-cat-section">\n'
        '      <label class="ed-cat-header">\n'
        '        <input type="checkbox" id="vis-verified" checked onchange="edToggleVis(\'verified\',this.checked)">\n'
        '        <span class="ed-swatch" style="background:#2563eb;opacity:0.7;"></span>\n'
        '        Verified (OSM + Foursquare)\n'
        '      </label>\n'
        '    </div>\n'
        '    <div class="ed-cat-section">\n'
        '      <label class="ed-cat-header">\n'
        '        <input type="checkbox" id="vis-osm_only" checked onchange="edToggleVis(\'osm_only\',this.checked)">\n'
        '        <span class="ed-swatch" style="background:#dc2626;opacity:0.7;"></span>\n'
        '        OSM Only\n'
        '      </label>\n'
        '    </div>\n'
        '    <div class="ed-cat-section">\n'
        '      <label class="ed-cat-header">\n'
        '        <input type="checkbox" id="vis-fsq_only" checked onchange="edToggleVis(\'fsq_only\',this.checked)">\n'
        '        <span class="ed-swatch" style="background:#dc2626;opacity:0.7;"></span>\n'
        '        Foursquare Only\n'
        '      </label>\n'
        '    </div>\n'
        '    <input type="text" id="ed-search" placeholder="Search golf course name..." oninput="edRenderList()">\n'
        '    <label style="display:flex;align-items:center;margin-top:6px;cursor:pointer;user-select:none;">\n'
        '      <input type="checkbox" id="ed-hover-toggle" checked onchange="edSetHoverPan(this.checked)" style="margin-right:6px;cursor:pointer;">\n'
        '      <span style="font-size:12px;color:#555;">Hover to pan (uncheck to require click)</span>\n'
        '    </label>\n'
        '  </div>\n'
        '  <div class="ed-count" id="ed-count"></div>\n'
        '  <div id="ed-facility-list"></div>\n'
        '  <button id="ed-export-btn" onclick="edExport()">Export GeoJSON</button>\n'
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
        '  var mapObj = null;\n'
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
        '  // --- Editing ---\n'
        '  function startEditing(id) {\n'
        '    if (editingId !== null) stopEditingInternal();\n'
        '    var layer = layers[id];\n'
        '    if (!layer) return;\n'
        '    editingId = id;\n'
        '    layer.eachLayer(function(sub) {\n'
        '      if (sub.pm) sub.pm.enable({allowSelfIntersection: false});\n'
        '      sub.setStyle({weight: 3, dashArray: "6,4"});\n'
        '    });\n'
        '    var f = facilityById(id);\n'
        '    var bar = document.getElementById("ed-edit-bar");\n'
        '    document.getElementById("ed-edit-name").textContent = f ? f.name : "";\n'
        '    bar.style.display = "flex";\n'
        '  }\n'
        '  function stopEditingInternal() {\n'
        '    if (editingId === null) return;\n'
        '    var layer = layers[editingId];\n'
        '    var f = facilityById(editingId);\n'
        '    if (layer) {\n'
        '      var color = (f && f.category === "verified") ? "#2563eb" : "#dc2626";\n'
        '      layer.eachLayer(function(sub) {\n'
        '        if (sub.pm) sub.pm.disable();\n'
        '        sub.setStyle({weight: 2, dashArray: null, color: color});\n'
        '      });\n'
        '    }\n'
        '    editingId = null;\n'
        '    document.getElementById("ed-edit-bar").style.display = "none";\n'
        '  }\n'
        '  window.edStopEditing = function() { stopEditingInternal(); };\n'
        '\n'
        '  // --- Context menu ---\n'
        '  function showCtxMenu(x, y, id) {\n'
        '    ctxTargetId = id;\n'
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
        '  window.edCtxEdit = function() {\n'
        '    var id = ctxTargetId;\n'
        '    hideCtxMenu();\n'
        '    if (id === null) return;\n'
        '    if (editingId === id) stopEditingInternal();\n'
        '    else startEditing(id);\n'
        '  };\n'
        '  window.edCtxDelete = function() {\n'
        '    var id = ctxTargetId;\n'
        '    hideCtxMenu();\n'
        '    if (id === null) return;\n'
        '    if (editingId === id) stopEditingInternal();\n'
        '    var m = getMap();\n'
        '    if (m && layers[id] && m.hasLayer(layers[id])) m.removeLayer(layers[id]);\n'
        '    deletedIds[id] = true;\n'
        '    fState[id].checked = false;\n'
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
        '  };\n'
        '\n'
        '  // --- Hover-to-pan toggle ---\n'
        '  window.edSetHoverPan = function(on) {\n'
        '    hoverPanEnabled = on;\n'
        '    edRenderList();\n'
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
        '        html += \'<span class="ed-fname"\' + hov + \' onclick="edZoomTo(\' + f.id + \')">\' + f.name + \'</span>\';\n'
        '        html += \'</div>\';\n'
        '        totalShown++;\n'
        '      }\n'
        '    }\n'
        '    list.innerHTML = html;\n'
        '    var checked = 0;\n'
        '    for (var i = 0; i < facilities.length; i++) { if (fState[facilities[i].id].checked) checked++; }\n'
        '    countEl.textContent = totalShown + " shown, " + checked + " checked for export";\n'
        '  };\n'
        '\n'
        '  // --- Export ---\n'
        '  window.edExport = function() {\n'
        '    var features = [];\n'
        '    for (var i = 0; i < facilities.length; i++) {\n'
        '      var f = facilities[i];\n'
        '      if (!fState[f.id].checked) continue;\n'
        '      var layer = layers[f.id];\n'
        '      if (!layer) continue;\n'
        '      var geom = null;\n'
        '      layer.eachLayer(function(sub) {\n'
        '        if (sub.toGeoJSON) geom = sub.toGeoJSON().geometry;\n'
        '      });\n'
        '      if (!geom) continue;\n'
        '      features.push({\n'
        '        type: "Feature",\n'
        '        properties: {\n'
        '          name: f.name,\n'
        '          category: f.category,\n'
        '          osm_id: f.osm_id,\n'
        '          fsq_id: f.fsq_id,\n'
        '          _node_ids: f._node_ids || null\n'
        '        },\n'
        '        geometry: geom\n'
        '      });\n'
        '    }\n'
        '    if (features.length === 0 && Object.keys(deletedIds).length === 0) { alert("No facilities checked for export."); return; }\n'
        '    var del = [];\n'
        '    for (var i = 0; i < facilities.length; i++) {\n'
        '      var df = facilities[i];\n'
        '      if (deletedIds[df.id]) del.push({osm_id: df.osm_id, fsq_id: df.fsq_id, name: df.name});\n'
        '    }\n'
        '    var fc = {type: "FeatureCollection", features: features, _deleted: del};\n'
        '    var blob = new Blob([JSON.stringify(fc, null, 2)], {type: "application/json"});\n'
        '    var url = URL.createObjectURL(blob);\n'
        '    var a = document.createElement("a");\n'
        '    a.href = url; a.download = "nj_golf_courses_edited.geojson";\n'
        '    document.body.appendChild(a); a.click();\n'
        '    document.body.removeChild(a); URL.revokeObjectURL(url);\n'
        '    alert("Exported " + features.length + " facilities to GeoJSON.");\n'
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
        '      // Build lookup indexes for matching\n'
        '      var byOsm = {}, byFsq = {}, byName = {};\n'
        '      for (var i = 0; i < facilities.length; i++) {\n'
        '        var f = facilities[i];\n'
        '        if (f.osm_id) byOsm[f.osm_id] = f;\n'
        '        if (f.fsq_id) byFsq[f.fsq_id] = f;\n'
        '        byName[f.name] = f;\n'
        '      }\n'
        '      var matched = 0, unmatched = 0;\n'
        '      var m = getMap();\n'
        '      // Reset: uncheck everything, clear deletions\n'
        '      for (var i = 0; i < facilities.length; i++) fState[facilities[i].id].checked = false;\n'
        '      deletedIds = {};\n'
        '      // Restore deletions from imported file\n'
        '      if (fc._deleted && fc._deleted.length) {\n'
        '        for (var d = 0; d < fc._deleted.length; d++) {\n'
        '          var dd = fc._deleted[d];\n'
        '          var dt = null;\n'
        '          if (dd.osm_id && byOsm[dd.osm_id]) dt = byOsm[dd.osm_id];\n'
        '          else if (dd.fsq_id && byFsq[dd.fsq_id]) dt = byFsq[dd.fsq_id];\n'
        '          else if (dd.name && byName[dd.name]) dt = byName[dd.name];\n'
        '          if (dt) {\n'
        '            deletedIds[dt.id] = true;\n'
        '            if (m && layers[dt.id] && m.hasLayer(layers[dt.id])) m.removeLayer(layers[dt.id]);\n'
        '          }\n'
        '        }\n'
        '      }\n'
        '      for (var j = 0; j < fc.features.length; j++) {\n'
        '        var feat = fc.features[j];\n'
        '        var p = feat.properties || {};\n'
        '        // Match by osm_id, then fsq_id, then name\n'
        '        var target = null;\n'
        '        if (p.osm_id && byOsm[p.osm_id]) target = byOsm[p.osm_id];\n'
        '        else if (p.fsq_id && byFsq[p.fsq_id]) target = byFsq[p.fsq_id];\n'
        '        else if (p.name && byName[p.name]) target = byName[p.name];\n'
        '        if (!target) { unmatched++; continue; }\n'
        '        matched++;\n'
        '        fState[target.id].checked = true;\n'
        '        // Replace layer geometry with imported (possibly edited) geometry\n'
        '        var oldLayer = layers[target.id];\n'
        '        if (oldLayer && m) {\n'
        '          var wasVisible = m.hasLayer(oldLayer);\n'
        '          if (wasVisible) m.removeLayer(oldLayer);\n'
        '          var color = (target.category === "verified") ? "#2563eb" : "#dc2626";\n'
        '          var newGJ = {type:"Feature", geometry: feat.geometry, properties:{id:target.id, name:target.name}};\n'
        '          var newLayer = L.geoJSON(newGJ, {\n'
        '            style: {fillColor: color, color: color, weight: 2, fillOpacity: 0.3}\n'
        '          });\n'
        '          newLayer.eachLayer(function(sub) {\n'
        '            sub.bindTooltip(target.name);\n'
        '            var tid = target.id;\n'
        '            sub.on("contextmenu", function(ev) {\n'
        '              L.DomEvent.preventDefault(ev);\n'
        '              showCtxMenu(ev.originalEvent.clientX, ev.originalEvent.clientY, tid);\n'
        '            });\n'
        '          });\n'
        '          layers[target.id] = newLayer;\n'
        '          if (wasVisible) newLayer.addTo(m);\n'
        '        }\n'
        '      }\n'
        '      edRenderList();\n'
        '      alert("Imported " + matched + " facilities."\n'
        '        + (unmatched > 0 ? " (" + unmatched + " could not be matched.)" : ""));\n'
        '    };\n'
        '    reader.readAsText(file);\n'
        '    input.value = "";  // reset so same file can be re-imported\n'
        '  };\n'
        '\n'
        '  // --- Keyboard shortcut ---\n'
        '  document.addEventListener("keydown", function(e) {\n'
        '    if (e.key === "Escape" && editingId !== null) stopEditingInternal();\n'
        '  });\n'
        '\n'
        '  // --- Init ---\n'
        '  setTimeout(function() {\n'
        '    initLayers();\n'
        '    edRenderList();\n'
        '  }, 800);\n'
        '})();\n'
        '</script>\n'
        '{% endmacro %}'
    ).replace("FACILITIES_JSON", facilities_json)

    sidebar = MacroElement()
    sidebar._template = Template(editable_tpl)
    m.get_root().add_child(sidebar)

    map_path = os.path.join(DATA_DIR, "nj_golf_courses_editable.html")
    m.save(map_path)
    print(f"  Editable map saved to {map_path}")
    return m


def create_tables(gdf, fsq_validation, web_validation=None, fsq_df=None):
    """Generate summary CSV table with POI and centroid coordinates."""
    print("[6/7] Generating summary tables...")

    gdf_projected = gdf.to_crs(UTM_CRS)
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

        # Web validation info
        web_validated = False
        course_type = "unknown"
        status = "unknown"
        if web_validation is not None and not web_validation.empty:
            web_match = web_validation[web_validation["name"] == name]
            if not web_match.empty:
                w = web_match.iloc[0]
                if w["course_type"] != "unknown" or w["status"] != "unknown":
                    web_validated = True
                course_type = w["course_type"]
                status = w["status"]

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
                "web_validated": web_validated,
                "course_type": course_type,
                "status": status,
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
                    "web_validated": False,
                    "course_type": "unknown",
                    "status": "unknown",
                }
            )

    summary_df = pd.DataFrame(records)
    csv_path = os.path.join(DATA_DIR, "nj_golf_courses_table.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved summary table ({len(summary_df)} courses) to {csv_path}")
    return summary_df


def export_to_josm(geojson_path, original_geojson_path=None):
    """Convert an edited GeoJSON file to .osm XML for import into JOSM.

    Reads the edited GeoJSON (exported from the editable map), compares it
    against the original to detect modifications, and produces a .osm file
    containing only changed/new features suitable for JOSM review and upload.

    Args:
        geojson_path: Path to the edited GeoJSON file.
        original_geojson_path: Path to the original GeoJSON for change detection.
            Defaults to data/nj_golf_courses.geojson.
    """
    import xml.etree.ElementTree as ET

    if original_geojson_path is None:
        original_geojson_path = os.path.join(DATA_DIR, "nj_golf_courses.geojson")

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
                ET.SubElement(way_el, "tag", k="leisure", v="golf_course")
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
    output_path = os.path.join(DATA_DIR, "nj_golf_courses_josm.osm")
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


def main():
    """Run the full pipeline."""
    print("=" * 60)
    print("NJ Golf Course Mapper & Validator")
    print("=" * 60)
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    errors = []

    # Step 1: Fetch OSM data
    try:
        osm_gdf = fetch_osm_golf_courses()
    except Exception as e:
        print(f"  ERROR fetching OSM data: {e}")
        errors.append(f"OSM fetch: {e}")
        return

    print()

    # Step 2: Fetch Foursquare data
    try:
        fsq_df = fetch_foursquare_golf_courses()
    except Exception as e:
        print(f"  ERROR fetching Foursquare data: {e}")
        errors.append(f"Foursquare fetch: {e}")
        fsq_df = pd.DataFrame(
            columns=["fsq_place_id", "name", "latitude", "longitude", "address", "locality"]
        )

    print()

    # Step 3: Cross-validate
    try:
        fsq_validation = validate_with_foursquare(osm_gdf, fsq_df)
    except Exception as e:
        print(f"  ERROR in Foursquare validation: {e}")
        errors.append(f"Foursquare validation: {e}")
        fsq_validation = pd.DataFrame()

    print()

    # Step 3b: Validate Foursquare-only POIs with land cover analysis (V1)
    try:
        fsq_validation = validate_foursquare_only(osm_gdf, fsq_validation)
    except Exception as e:
        print(f"  ERROR in Foursquare-only V1 validation: {e}")
        errors.append(f"Foursquare-only V1 validation: {e}")

    print()

    # Step 3c: V2 building-based directional ellipse validation
    try:
        fsq_validation = validate_foursquare_only_v2(osm_gdf, fsq_validation, fsq_df)
    except Exception as e:
        print(f"  ERROR in Foursquare-only V2 validation: {e}")
        errors.append(f"Foursquare-only V2 validation: {e}")

    print()

    # Step 4: Web validation
    try:
        web_validation = validate_with_web(osm_gdf)
    except Exception as e:
        print(f"  ERROR in web validation: {e}")
        errors.append(f"Web validation: {e}")
        web_validation = None

    print()

    # Step 5: Create map
    try:
        create_map(osm_gdf, fsq_validation, web_validation)
    except Exception as e:
        print(f"  ERROR creating map: {e}")
        errors.append(f"Map creation: {e}")

    print()

    # Step 5b: Create editable map
    try:
        create_editable_map(osm_gdf, fsq_validation)
    except Exception as e:
        print(f"  ERROR creating editable map: {e}")
        errors.append(f"Editable map creation: {e}")

    print()

    # Step 6 & 7: Create summary tables
    try:
        summary = create_tables(osm_gdf, fsq_validation, web_validation, fsq_df)
    except Exception as e:
        print(f"  ERROR creating tables: {e}")
        errors.append(f"Table creation: {e}")
        summary = None

    print()

    # Final summary
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total golf courses found (OSM): {len(osm_gdf)}")
    print(f"Foursquare courses found: {len(fsq_df)}")

    if not fsq_validation.empty:
        matched = len(fsq_validation[fsq_validation["foursquare_match"] == True])
        total = len(fsq_validation[fsq_validation["osm_id"] != ""])
        if total > 0:
            print(f"Foursquare match rate: {matched}/{total} ({matched/total*100:.1f}%)")
        if "golf_probability" in fsq_validation.columns:
            fsq_only = fsq_validation[fsq_validation["match_method"] == "foursquare_only"]
            likely = fsq_only[fsq_only["golf_probability"] >= 0.5]
            print(f"Foursquare-only likely golf courses (V1 land cover): {len(likely)}/{len(fsq_only)}")
        if "golf_probability_v2" in fsq_validation.columns:
            fsq_only = fsq_validation[fsq_validation["match_method"] == "foursquare_only"]
            likely_v2 = fsq_only[fsq_only["golf_probability_v2"] >= 0.5]
            print(f"Foursquare-only likely golf courses (V2 building):   {len(likely_v2)}/{len(fsq_only)}")

    if summary is not None:
        web_val = summary[summary["web_validated"] == True]
        print(f"Web-validated courses: {len(web_val)}")

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
            edited_path = os.path.join(DATA_DIR, "nj_golf_courses_edited.geojson")
        original_path = None
        if "--original" in sys.argv:
            oi = sys.argv.index("--original")
            if oi + 1 < len(sys.argv):
                original_path = sys.argv[oi + 1]
        export_to_josm(edited_path, original_path)
    else:
        main()
