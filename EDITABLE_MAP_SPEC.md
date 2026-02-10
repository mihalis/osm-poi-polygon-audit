# Editable Polygon Map — Reusable Specification

This document describes the architecture, features, data contracts, and implementation
details of the editable polygon map built for NJ golf courses. It is written so that
the system can be reproduced for **any facility type** (parking lots, parks, buildings,
etc.) by adapting the data preparation layer while keeping the front-end template intact.

---

## 1. High-Level Architecture

```
Python (data prep)                   Browser (HTML/JS)
──────────────────                   ─────────────────
GeoDataFrame / CSV                   Folium base map
        │                                   │
        ▼                                   ▼
Build facilities JSON ──────────►  Embedded JSON array
(simplified geometries,            parsed on page load
 categories, metadata)                     │
        │                                   ▼
        ▼                            Leaflet layers +
Folium map + MacroElement            Leaflet-Geoman
injects sidebar + JS into               │      │
a single self-contained            Sidebar   Context menu
HTML file (~650-850 KB)            (list,    (edit / delete)
                                   search,       │
                                   filters)       ▼
                                       │     Export / Import
                                       ▼     GeoJSON files
                                   Checkboxes
                                   (export selection)
```

**Key principle:** Everything lives in a single `.html` file with zero server
dependencies. The file can be opened from disk (`file://`) or served statically.

---

## 2. Python-Side Data Preparation

### 2.1 Inputs

| Input | Type | Purpose |
|-------|------|---------|
| `gdf` | GeoDataFrame (EPSG:4326) | Polygons with geometry + attributes |
| `validation_df` | pandas DataFrame | Cross-reference table mapping sources (optional) |

For a generic app, `validation_df` may not exist. The minimum requirement is a
GeoDataFrame with a `geometry` column and a `name` column.

### 2.2 Polygon Simplification

Simplification is applied **before** serializing to JSON to reduce file size and
improve browser rendering performance.

```python
from shapely.validation import make_valid
from shapely.geometry import MultiPolygon

gdf_projected = gdf.to_crs("EPSG:32618")  # UTM zone — use appropriate zone

# Fix invalid geometries, then simplify
gdf_simple = gdf_projected.copy()
gdf_simple["geometry"] = gdf_simple.geometry.apply(
    lambda g: make_valid(g) if not g.is_valid else g
).simplify(tolerance=10, preserve_topology=True)  # 10 meters

# Simplification can produce GeometryCollections; extract polygon parts
def _extract_polygons(geom):
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
        if len(polys) == 1:
            return polys[0]
        elif polys:
            return MultiPolygon(polys)
    return geom

gdf_simple["geometry"] = gdf_simple.geometry.apply(_extract_polygons)
gdf_simple = gdf_simple.to_crs("EPSG:4326")  # back to WGS84
```

**Tolerance tuning:**
- `10` meters works well for large features (golf courses, parks, campuses)
- Use `3`-`5` for smaller features (parking lots, buildings)
- Use `20`+ for very large features (forests, lakes)

The NJ golf course dataset went from 22,577 to 13,040 vertices (42% reduction).

### 2.3 Hexagon Generation for Point-Only Data

When a data source provides only a point (lat/lon) with no polygon, a placeholder
hexagon is generated so the user can drag its vertices into the correct shape.

```python
import math

def _make_hexagon(lat, lon, radius_m=75):
    """Generate a flat-top hexagon polygon (GeoJSON dict) centered on lat/lon."""
    lat_off = radius_m / 111000
    lon_off = radius_m / (111000 * math.cos(math.radians(lat)))
    coords = []
    for i in range(6):
        angle = math.radians(60 * i - 30)  # flat-top orientation
        coords.append([
            lon + lon_off * math.cos(angle),
            lat + lat_off * math.sin(angle)
        ])
    coords.append(coords[0])  # close the ring
    return {"type": "Polygon", "coordinates": [coords]}
```

- `radius_m=75` gives a ~150m-wide hexagon — a reasonable starting size for a
  golf course. Use `25`-`50` for parking lots.
- The formula accounts for latitude distortion (longitude degrees are narrower
  at higher latitudes).

### 2.4 Facilities JSON Schema

The Python side builds an array of facility objects and embeds it as a JSON literal
inside the HTML. Each object:

```json
{
  "id": 0,
  "name": "Pine Valley Golf Club",
  "category": "verified",
  "checked": true,
  "geometry": { "type": "Polygon", "coordinates": [[[...], ...]] },
  "osm_id": "way/12345",
  "fsq_id": "4b5e..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | integer | yes | Unique sequential ID (0-based). Used as key in all JS lookups. |
| `name` | string | yes | Display name shown in sidebar and tooltip. |
| `category` | string | yes | Grouping key. Determines color and sidebar section. |
| `checked` | boolean | yes | Initial export-checkbox state. `true` = pre-selected for export. |
| `geometry` | GeoJSON Geometry | yes | Polygon or MultiPolygon. Serialized via `.__geo_interface__`. |
| `osm_id` | string | no | Source identifier #1. Used for matching on import. |
| `fsq_id` | string | no | Source identifier #2. Used for matching on import. |

**Adapting for other projects:** Rename `osm_id`/`fsq_id` to whatever source
identifiers make sense (e.g., `parcel_id`, `lot_number`). The import matching
logic tries fields in order: first source ID, then second source ID, then name.
Add or remove ID fields as needed — just update the export properties block and
the import matching block in the JS template to match.

### 2.5 Categories

Categories are fully configurable. In the NJ golf course app, three are used:

| Category key | Color | Default checked | Source |
|-------------|-------|-----------------|--------|
| `verified` | `#2563eb` (blue) | `true` | Matched in both OSM and Foursquare |
| `osm_only` | `#dc2626` (red) | `false` | OSM polygon, no Foursquare match |
| `fsq_only` | `#dc2626` (red) | `false` | Foursquare point only (hexagon placeholder) |

**To customize:**
1. Change the `category` values assigned during data preparation
2. Update the three places in the JS/HTML template that reference categories:
   - The `cats` array in `edRenderList` (sidebar section headers)
   - The category visibility checkboxes in the sidebar HTML
   - The color assignment: `(f.category === "verified") ? "#2563eb" : "#dc2626"`

For a two-category system (e.g., "confirmed" vs. "unconfirmed"), remove the third
section and adjust the color logic. For a single category, remove all category
filtering entirely.

---

## 3. HTML/JS Front-End — Feature Reference

### 3.1 Base Map

- **Library:** Folium (Python) generating a Leaflet map
- **Center:** Mean centroid of all input geometries
- **Zoom:** 8 (state-level for NJ; adjust for your area)
- **Tile layers** (via `folium.TileLayer`):
  - OpenStreetMap (default)
  - Google Roads
  - Google Satellite
  - Mapbox Streets / Satellite (if `MAPBOX_TOKEN` is set)
- **Layer control:** Collapsed, top-right corner

### 3.2 Leaflet-Geoman Integration

Loaded from unpkg CDN (no local files needed):
```html
<link rel="stylesheet" href="https://unpkg.com/@geoman-io/leaflet-geoman-free@latest/dist/leaflet-geoman.css" />
<script src="https://unpkg.com/@geoman-io/leaflet-geoman-free@latest/dist/leaflet-geoman.min.js"></script>
```

Geoman provides per-layer vertex editing via `layer.pm.enable()` / `.pm.disable()`.
It is **not** enabled globally — editing is activated only on the specific polygon
the user right-clicks.

### 3.3 Sidebar (Left Panel, 340px wide)

The sidebar is a fixed-position `div` that pushes the map to the right via:
```css
.folium-map { margin-left: 340px !important; width: calc(100% - 340px) !important; }
```

#### 3.3.1 Header Section

Contains, from top to bottom:

1. **Title** — `<h3>` with the app name
2. **Hint text** — Single-line description of the interaction model
3. **Category filter checkboxes** — One per category. Controls:
   - **Map visibility:** checking/unchecking shows/hides all polygons in that category
   - **Sidebar filtering:** the facility list only shows categories that are checked
4. **Search bar** — Text input that filters the facility list by name substring
   (case-insensitive). Filtering is instant (on every keystroke via `oninput`).
5. **Hover-to-pan toggle** — Checkbox controlling whether hovering over a name
   in the list pans the map to that facility.

#### 3.3.2 Facility List

- Scrollable container (`overflow-y: auto`) filling remaining sidebar height
- Grouped by category with uppercase section headers showing count, e.g.,
  `VERIFIED (OSM + FOURSQUARE) (290)`
- Each facility row contains:
  - **Checkbox** (left) — Controls export inclusion only. Does NOT affect map visibility.
    Title attribute: "Include in export"
  - **Color dot** — 9px circle matching category color
  - **Name** (click to zoom, hover to pan if toggle is on)

#### 3.3.3 Status Bar

Below the header, above the list:
```
345 shown, 290 checked for export
```
Updates live as filters and checkboxes change.

#### 3.3.4 Buttons

At the bottom of the sidebar, two buttons:

1. **Export GeoJSON** (blue, filled) — Downloads checked facilities as GeoJSON
2. **Import GeoJSON** (blue, outlined) — Opens a file picker to load a previous export

### 3.4 Context Menu (Right-Click on Polygon)

A custom context menu replaces the browser default on polygon right-click:

```
┌──────────────┐
│ Edit Shape   │
│ Delete       │  ← red on hover
└──────────────┘
```

- Positioned at the mouse cursor (`clientX`/`clientY` from the DOM event)
- Dismissed by clicking anywhere else on the page
- Only one can be open at a time

### 3.5 Edit Mode

Triggered by: right-click polygon > "Edit Shape"

**Entering edit mode:**
1. Any previously-editing polygon is finished first (single-edit constraint)
2. `layer.pm.enable({allowSelfIntersection: false})` is called on the target
3. Polygon border changes to `weight: 3, dashArray: "6,4"` (thick dashed line)
4. A yellow notification bar appears at the top of the map:
   ```
   Editing: Pine Valley Golf Club (drag vertices to reshape — right-click or Esc to finish) [Done]
   ```

**While editing:**
- Vertices appear as draggable white circles (Geoman default)
- User can drag vertices, and Geoman handles midpoint insertion automatically
- The polygon reshapes in real-time

**Exiting edit mode** (three methods):
1. Right-click the same polygon again > "Edit Shape" (toggles off)
2. Press **Escape** key
3. Click the **Done** button in the notification bar

**On exit:**
- `layer.pm.disable()` is called
- Border reverts to `weight: 2, dashArray: null`
- Notification bar hides
- The edited geometry is preserved in the Leaflet layer object

### 3.6 Delete

Triggered by: right-click polygon > "Delete"

**What happens:**
1. If the polygon is being edited, editing is stopped first
2. The layer is removed from the map (`map.removeLayer`)
3. The facility ID is added to `deletedIds` (a JS object used as a set)
4. The facility's checked state is set to `false`
5. `edRenderList()` is called — the facility disappears from the sidebar

**Deleted facilities are invisible.** They don't appear in the sidebar list,
they don't appear on the map, and they are not included in export.

**Undo is not supported** within a session. However, reloading the page restores
all facilities to their original state. Deletions are only persisted through
export/import.

### 3.7 Export

**Button:** "Export GeoJSON" at the bottom of the sidebar.

**Process:**
1. Iterates all facilities
2. For each facility where `fState[id].checked === true`:
   - Reads the **current** layer geometry via `layer.toGeoJSON()` — this captures
     any vertex edits the user made
   - Builds a GeoJSON Feature with properties: `name`, `category`, `osm_id`, `fsq_id`
3. Collects all deleted facility identifiers (`osm_id`, `fsq_id`, `name`)
4. Builds a GeoJSON FeatureCollection with a custom `_deleted` array
5. Triggers a browser download as `nj_golf_courses_edited.geojson`
6. Shows an alert with the count

**Output format:**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "Pine Valley Golf Club",
        "category": "verified",
        "osm_id": "way/12345",
        "fsq_id": "4b5e..."
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[...], ...]]
      }
    }
  ],
  "_deleted": [
    {"osm_id": "way/99999", "fsq_id": "", "name": "Defunct Golf Course"}
  ]
}
```

The `_deleted` array is a custom extension at the FeatureCollection level. Standard
GeoJSON parsers will ignore it. The editable map's import function reads it.

### 3.8 Import

**Button:** "Import GeoJSON" at the bottom of the sidebar (opens a file picker).

**Process:**
1. Reads the selected `.geojson` or `.json` file as text
2. Parses JSON; validates that `features` array exists
3. Builds lookup indexes from the in-memory facilities: `byOsm`, `byFsq`, `byName`
4. **Resets state:** unchecks all facilities, clears all deletions
5. **Restores deletions:** reads `_deleted` array, matches each to a facility,
   removes its layer from the map, marks it as deleted
6. **Restores features:** for each imported feature:
   - Matches to a facility by `osm_id` → `fsq_id` → `name` (in priority order)
   - Sets `checked = true`
   - **Replaces the map layer** with the imported geometry (preserving edits):
     - Removes old layer
     - Creates new `L.geoJSON` with imported geometry
     - Re-attaches tooltip and right-click handler
     - Adds to map if category is visible
7. Re-renders the sidebar list
8. Shows alert: "Imported 150 facilities. (2 could not be matched.)"

**Matching logic priority:**
1. `osm_id` (exact match)
2. `fsq_id` (exact match)
3. `name` (exact match, last resort)

If a feature cannot be matched to any facility, it is counted as "unmatched"
and reported in the alert. This can happen if the base data has changed since
the export was created.

---

## 4. JavaScript State Model

All state lives inside an IIFE (immediately-invoked function expression) to
avoid polluting the global scope. Functions exposed to HTML `onclick` handlers
are attached to `window`.

### 4.1 Variables

| Variable | Type | Description |
|----------|------|-------------|
| `facilities` | Array | The embedded JSON array (read-only after init) |
| `layers` | Object `{id: L.geoJSON}` | Leaflet layer group for each facility |
| `fState` | Object `{id: {checked: bool}}` | Export checkbox state |
| `catVisible` | Object `{category: bool}` | Category visibility toggles |
| `editingId` | int or null | ID of the polygon currently being edited |
| `hoverPanEnabled` | bool | Whether hovering over sidebar names pans the map |
| `deletedIds` | Object `{id: true}` | Set of deleted facility IDs |
| `ctxTargetId` | int or null | ID of the polygon the context menu was opened on |
| `mapObj` | L.Map or null | Cached reference to the Folium-generated Leaflet map |

### 4.2 Functions

| Function | Scope | Trigger | Description |
|----------|-------|---------|-------------|
| `getMap()` | private | internal | Finds the Leaflet map instance by scanning `window` for objects with `_leaflet_id` and `getCenter` |
| `facilityById(id)` | private | internal | Linear scan of `facilities` array |
| `startEditing(id)` | private | context menu | Enables Geoman on target layer, shows edit bar |
| `stopEditingInternal()` | private | multiple | Disables Geoman, restores style, hides edit bar |
| `showCtxMenu(x, y, id)` | private | right-click | Positions and shows the context menu div |
| `hideCtxMenu()` | private | document click | Hides context menu |
| `initLayers()` | private | init | Creates all L.geoJSON layers, attaches tooltips and right-click handlers, adds to map |
| `showLayer(id)` | private | internal | Adds a layer to the map if not already present |
| `hideLayer(id)` | private | internal | Removes a layer from the map if present |
| `edToggleVis(cat, on)` | window | category checkbox | Shows/hides all layers in a category, re-renders sidebar |
| `edToggleItem(id, on)` | window | facility checkbox | Sets checked state (export-only, no visibility change) |
| `edSetHoverPan(on)` | window | hover toggle | Sets `hoverPanEnabled`, re-renders list |
| `edZoomTo(id)` | window | click on name | `map.fitBounds` with `maxZoom: 16, padding: [40, 40]` |
| `edPanTo(id)` | window | hover on name | `map.panTo` (center of bounds) |
| `edRenderList()` | window | many triggers | Rebuilds the entire sidebar facility list HTML |
| `edExport()` | window | export button | Collects checked features + deleted list, triggers download |
| `edImport(input)` | window | file input change | Reads file, matches features, replaces layers, restores state |
| `edCtxEdit()` | window | context menu | Calls `startEditing` on the context-menu target |
| `edCtxDelete()` | window | context menu | Removes layer, marks as deleted, re-renders |
| `edStopEditing()` | window | Done button | Delegates to `stopEditingInternal()` |

### 4.3 Event Listeners

| Event | Target | Action |
|-------|--------|--------|
| `contextmenu` | Each polygon sublayer | `L.DomEvent.preventDefault` + `showCtxMenu` |
| `click` | `document` | `hideCtxMenu()` (dismiss context menu) |
| `keydown` | `document` | If `Escape` and editing, call `stopEditingInternal()` |

---

## 5. CSS Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ [ed-edit-bar — yellow, fixed, top, z:1002, only visible when editing]
│ [ed-ctx-menu — white, fixed, z:2000, only visible on right-click]
├──────────────┬──────────────────────────────────────────────────┤
│  #ed-sidebar │                                                  │
│  fixed left  │              .folium-map                         │
│  340px wide  │              margin-left: 340px                  │
│  z: 1001     │              width: calc(100% - 340px)           │
│              │                                                  │
│  ┌────────┐  │         Leaflet map fills remaining space        │
│  │ header │  │                                                  │
│  │ search │  │                                                  │
│  │ toggle │  │                                                  │
│  ├────────┤  │                                                  │
│  │ count  │  │                                                  │
│  ├────────┤  │                                                  │
│  │        │  │                                                  │
│  │  list  │  │                                                  │
│  │(scroll)│  │                                                  │
│  │        │  │                                                  │
│  ├────────┤  │                                                  │
│  │ export │  │                                                  │
│  │ import │  │                                                  │
│  └────────┘  │                                                  │
└──────────────┴──────────────────────────────────────────────────┘
```

### Z-index layers
- `1001` — Sidebar (above map controls)
- `1002` — Edit notification bar
- `2000` — Context menu (above everything)

---

## 6. Adapting for a New Project

### 6.1 Minimum Changes Checklist

| # | What to change | Where |
|---|---------------|-------|
| 1 | Input data (GeoDataFrame) | Python: data loading before `create_editable_map` |
| 2 | Category definitions | Python: the logic that assigns `category` to each facility |
| 3 | Category names & colors | HTML template: sidebar checkboxes, JS: color logic, `cats` array in `edRenderList` |
| 4 | Source ID fields | Python: the fields added to each facility dict; JS: export properties block, import matching block |
| 5 | Simplification tolerance | Python: `.simplify(tolerance)` — lower for smaller features |
| 6 | Hexagon radius | Python: `_make_hexagon(lat, lon, radius_m=...)` — smaller for smaller features |
| 7 | Map center & zoom | Python: `folium.Map(location=..., zoom_start=...)` |
| 8 | Output filename | Python: `map_path` variable; JS: `a.download` in export function |
| 9 | Title & hint text | HTML template: `<h3>` and `.ed-hint` |
| 10 | UTM zone | Python: `UTM_CRS` constant — must match your geographic region |

### 6.2 Single-Category Simplification

If you only have one data source (no verified/unverified distinction):

1. Set all facilities to `category: "all"` with a single color
2. Remove the two extra category checkboxes from the sidebar HTML
3. Simplify the color logic to just one color
4. Remove the `catVisible` filtering (or keep it with one always-on toggle)

### 6.3 Adding More Categories

1. Add the category checkbox HTML block (copy an existing one, change the `id`,
   label, and swatch color)
2. Add the category to the `catVisible` initial state object
3. Add the category to the `cats` array in `edRenderList`
4. Update the color logic (currently a ternary; switch to a lookup object):
   ```js
   var colorMap = {verified: "#2563eb", osm_only: "#dc2626", fsq_only: "#dc2626"};
   var color = colorMap[f.category] || "#666";
   ```

### 6.4 Adding Custom Properties

To add more metadata fields per facility (e.g., `address`, `area_sqm`, `source_url`):

1. Python: add the field to each facility dict
2. JS export: add it to the `properties` block in `edExport`
3. JS import: no change needed (extra properties are preserved through matching)
4. Optionally show it in a popup or tooltip by modifying the `bindTooltip` call

### 6.5 Extracting as a Standalone Module

The `create_editable_map` function and its helpers (`_make_hexagon`,
`_count_polygon_vertices`) can be extracted into a standalone Python module.
The function signature would become:

```python
def create_editable_map(
    gdf,                          # GeoDataFrame with geometry + name
    output_path,                  # Where to save the HTML
    categories,                   # List of {key, label, color, default_checked}
    category_column="category",   # Column in gdf for category assignment
    source_id_fields=None,        # e.g., ["parcel_id", "source_ref"]
    simplify_tolerance_m=10,      # Shapely simplify tolerance in meters
    hexagon_points=None,          # Optional DataFrame of point-only entries
    hexagon_radius_m=75,          # Hexagon placeholder radius
    map_center=None,              # [lat, lon] or None for auto
    map_zoom=8,
    title="Editable Map",
    utm_crs="EPSG:32618",
):
```

---

## 7. Known Limitations & Future Improvements

### Current limitations

- **No undo** for deletes within a session (reload to reset, or re-import)
- **No undo** for vertex edits (Leaflet-Geoman limitation in the free version)
- **Single-edit only** — one polygon at a time can be in edit mode
- **Name matching on import** is exact — typos or name changes will cause
  mismatches (falls through to "unmatched")
- **Large datasets** (5,000+ polygons) may cause slow initial render. Consider
  pre-filtering or using Leaflet.markercluster for the initial view.
- **GeometryCollections** are normalized to Polygon/MultiPolygon during
  simplification, but if the source data contains lines or points mixed with
  polygons, those will be silently dropped.

### Possible enhancements

- **Undo stack** — track geometry snapshots before each edit and allow Ctrl+Z
- **Bulk operations** — select multiple polygons and delete/check/uncheck all
- **Keyboard navigation** — arrow keys to step through the facility list
- **Progress persistence** — auto-save to `localStorage` so work survives
  accidental page refresh without manual export
- **Split view** — show satellite imagery side-by-side with the editable map
- **Area calculation** — display polygon area in the sidebar or tooltip
- **Validation** — highlight self-intersecting or zero-area polygons
- **Draw new polygons** — use Geoman's draw tools to create polygons not in
  the original dataset

---

## 8. File Reference

| File | Purpose |
|------|---------|
| `nj_golf_courses.py` | Main script. Contains `create_editable_map()` plus all data fetching/processing. |
| `data/nj_golf_courses_editable.html` | Generated output. Self-contained, no server needed. |
| `data/nj_golf_courses_edited.geojson` | User-exported file. Contains checked features + `_deleted` array. |
| `EDITABLE_MAP_SPEC.md` | This document. |

---

## 9. Dependency Versions

| Dependency | Version used | Notes |
|-----------|-------------|-------|
| Python | 3.9+ | f-strings, walrus operator not used |
| folium | 0.14+ | `MacroElement`, `Template` for custom HTML injection |
| geopandas | 0.12+ | `.to_crs()`, `.simplify()` |
| shapely | 1.8+ / 2.0+ | `make_valid`, `__geo_interface__` |
| pandas | 1.4+ | Standard DataFrame operations |
| Leaflet | (bundled by folium) | ~1.9 |
| Leaflet-Geoman Free | latest via unpkg CDN | `pm.enable()` / `pm.disable()` |

No npm, no build step, no bundler. The CDN-loaded Leaflet-Geoman is the only
runtime dependency beyond what Folium bundles.
