"""Tests for nj_golf_courses.py — _make_hexagon, _count_polygon_vertices, export_to_josm."""

import json
import math
import os
import xml.etree.ElementTree as ET

import pytest

import nj_golf_courses as mod


# ---------------------------------------------------------------------------
# _make_hexagon tests
# ---------------------------------------------------------------------------

class TestMakeHexagon:
    def test_closed_ring_of_7_coords(self):
        h = mod._make_hexagon(40.0, -74.5)
        ring = h["coordinates"][0]
        assert len(ring) == 7, "Hexagon ring should have 7 coordinates (6 vertices + closing)"
        assert ring[0] == ring[-1], "First and last coordinates must match (closed ring)"

    def test_geometry_type_is_polygon_with_single_ring(self):
        h = mod._make_hexagon(40.0, -74.5)
        assert h["type"] == "Polygon"
        assert len(h["coordinates"]) == 1, "Should have exactly one ring"

    def test_centroid_near_input(self):
        lat, lon = 40.0, -74.5
        h = mod._make_hexagon(lat, lon)
        ring = h["coordinates"][0][:-1]  # exclude closing vertex
        avg_lon = sum(c[0] for c in ring) / len(ring)
        avg_lat = sum(c[1] for c in ring) / len(ring)
        assert abs(avg_lat - lat) < 0.001
        assert abs(avg_lon - lon) < 0.001

    def test_larger_radius_larger_hexagon(self):
        h_small = mod._make_hexagon(40.0, -74.5, radius_m=50)
        h_large = mod._make_hexagon(40.0, -74.5, radius_m=200)

        def ring_span(h):
            ring = h["coordinates"][0][:-1]
            lons = [c[0] for c in ring]
            lats = [c[1] for c in ring]
            return max(lons) - min(lons), max(lats) - min(lats)

        s_lon, s_lat = ring_span(h_small)
        l_lon, l_lat = ring_span(h_large)
        assert l_lon > s_lon
        assert l_lat > s_lat

    def test_higher_latitude_wider_longitude_spread(self):
        """At higher latitudes, the same radius in meters should produce a wider
        longitude spread because degrees of longitude are smaller."""
        h_low = mod._make_hexagon(30.0, -74.5, radius_m=100)
        h_high = mod._make_hexagon(60.0, -74.5, radius_m=100)

        def lon_spread(h):
            ring = h["coordinates"][0][:-1]
            lons = [c[0] for c in ring]
            return max(lons) - min(lons)

        assert lon_spread(h_high) > lon_spread(h_low)


# ---------------------------------------------------------------------------
# _count_polygon_vertices tests
# ---------------------------------------------------------------------------

class TestCountPolygonVertices:
    def test_simple_polygon(self):
        geom = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
        assert mod._count_polygon_vertices(geom) == 4

    def test_polygon_with_holes(self):
        outer = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
        hole = [[2, 2], [8, 2], [8, 8], [2, 2]]
        geom = {"type": "Polygon", "coordinates": [outer, hole]}
        assert mod._count_polygon_vertices(geom) == len(outer) + len(hole)

    def test_multipolygon(self):
        poly1 = [[[0, 0], [1, 0], [1, 1], [0, 0]]]
        poly2 = [[[2, 2], [3, 2], [3, 3], [2, 2]]]
        geom = {"type": "MultiPolygon", "coordinates": [poly1, poly2]}
        assert mod._count_polygon_vertices(geom) == 4 + 4

    def test_geometry_collection(self):
        inner = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
        geom = {"type": "GeometryCollection", "geometries": [inner]}
        assert mod._count_polygon_vertices(geom) == 4

    def test_non_polygon_returns_zero(self):
        assert mod._count_polygon_vertices({"type": "Point", "coordinates": [0, 0]}) == 0
        assert mod._count_polygon_vertices({"type": "LineString", "coordinates": [[0, 0], [1, 1]]}) == 0


# ---------------------------------------------------------------------------
# export_to_josm tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def josm_env(tmp_path, monkeypatch):
    """Set up DATA_DIR to tmp_path and provide helper to write GeoJSON files."""
    monkeypatch.setattr(mod, "DATA_DIR", str(tmp_path))

    def write_geojson(name, features, deleted=None):
        fc = {"type": "FeatureCollection", "features": features}
        if deleted is not None:
            fc["_deleted"] = deleted
        path = tmp_path / name
        path.write_text(json.dumps(fc))
        return str(path)

    return tmp_path, write_geojson


class TestExportToJosm:
    def _parse_osm(self, path):
        return ET.parse(path).getroot()

    def test_unchanged_features_skipped(self, josm_env):
        tmp_path, write_gj = josm_env
        feat = {
            "type": "Feature",
            "properties": {"name": "Test", "osm_id": "way/100", "osm_type": "way",
                           "_node_ids": [[1, 2, 3, 1]]},
            "geometry": {"type": "Polygon",
                         "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        }
        write_gj("nj_golf_courses_data.geojson", [feat])
        edited_path = write_gj("edited.geojson", [feat])

        out = mod.export_to_josm(edited_path)
        root = self._parse_osm(out)
        ways = root.findall("way")
        assert len(ways) == 0, "Unchanged features should produce 0 ways"

    def test_modified_way(self, josm_env):
        tmp_path, write_gj = josm_env
        orig_feat = {
            "type": "Feature",
            "properties": {"name": "Test", "osm_id": "way/100", "osm_type": "way",
                           "_node_ids": [[1, 2, 3, 1]]},
            "geometry": {"type": "Polygon",
                         "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        }
        # Shift one vertex
        edited_feat = json.loads(json.dumps(orig_feat))
        edited_feat["geometry"]["coordinates"][0][1] = [1.5, 0]
        write_gj("nj_golf_courses_data.geojson", [orig_feat])
        edited_path = write_gj("edited.geojson", [edited_feat])

        out = mod.export_to_josm(edited_path)
        root = self._parse_osm(out)
        ways = root.findall("way")
        assert len(ways) == 1
        way = ways[0]
        assert way.attrib["id"] == "100"
        assert way.attrib["action"] == "modify"
        # Nodes should have action="modify" and positive IDs
        nodes = root.findall("node")
        assert all(int(n.attrib["id"]) > 0 for n in nodes)
        assert all(n.attrib.get("action") == "modify" for n in nodes)
        # Way should have nd refs matching node ids
        nd_refs = [nd.attrib["ref"] for nd in way.findall("nd")]
        node_ids = [n.attrib["id"] for n in nodes]
        for ref in nd_refs:
            assert ref in node_ids

    def test_new_polygon_negative_ids(self, josm_env):
        tmp_path, write_gj = josm_env
        write_gj("nj_golf_courses_data.geojson", [])  # empty original
        new_feat = {
            "type": "Feature",
            "properties": {"name": "New Course", "osm_id": "", "osm_type": ""},
            "geometry": {"type": "Polygon",
                         "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        }
        edited_path = write_gj("edited.geojson", [new_feat])

        out = mod.export_to_josm(edited_path)
        root = self._parse_osm(out)
        ways = root.findall("way")
        assert len(ways) == 1
        way = ways[0]
        assert int(way.attrib["id"]) < 0
        nodes = root.findall("node")
        assert all(int(n.attrib["id"]) < 0 for n in nodes)
        # Should have leisure=golf_course, name, source tags
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
        assert tags["leisure"] == "golf_course"
        assert tags["name"] == "New Course"
        assert "source" in tags

    def test_modified_relation_warns(self, josm_env, capsys):
        tmp_path, write_gj = josm_env
        orig_feat = {
            "type": "Feature",
            "properties": {"name": "Rel Course", "osm_id": "relation/200",
                           "osm_type": "relation", "_node_ids": [[10, 11, 12, 10]]},
            "geometry": {"type": "Polygon",
                         "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        }
        edited_feat = json.loads(json.dumps(orig_feat))
        edited_feat["geometry"]["coordinates"][0][1] = [1.5, 0]
        write_gj("nj_golf_courses_data.geojson", [orig_feat])
        edited_path = write_gj("edited.geojson", [edited_feat])

        mod.export_to_josm(edited_path)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "relation/200" in captured.out

    def test_tag_preservation(self, josm_env):
        tmp_path, write_gj = josm_env
        orig_feat = {
            "type": "Feature",
            "properties": {"name": "Tag Course", "osm_id": "way/300",
                           "osm_type": "way", "_node_ids": [[1, 2, 3, 1]],
                           "leisure": "golf_course", "access": "private"},
            "geometry": {"type": "Polygon",
                         "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        }
        edited_feat = json.loads(json.dumps(orig_feat))
        edited_feat["geometry"]["coordinates"][0][1] = [1.5, 0]
        write_gj("nj_golf_courses_data.geojson", [orig_feat])
        edited_path = write_gj("edited.geojson", [edited_feat])

        out = mod.export_to_josm(edited_path)
        root = self._parse_osm(out)
        way = root.findall("way")[0]
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
        assert tags.get("leisure") == "golf_course"
        assert tags.get("access") == "private"
        assert tags.get("name") == "Tag Course"

    def test_output_is_valid_xml(self, josm_env):
        tmp_path, write_gj = josm_env
        write_gj("nj_golf_courses_data.geojson", [])
        new_feat = {
            "type": "Feature",
            "properties": {"name": "XML Course", "osm_id": ""},
            "geometry": {"type": "Polygon",
                         "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        }
        edited_path = write_gj("edited.geojson", [new_feat])

        out = mod.export_to_josm(edited_path)
        root = self._parse_osm(out)
        assert root.tag == "osm"
        assert root.attrib["version"] == "0.6"
