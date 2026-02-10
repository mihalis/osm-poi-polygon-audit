# NJ Golf Courses — Editable Map User Guide

## Getting Started

Open the file `nj_golf_courses_editable.html` in a modern browser (Chrome, Firefox, or Edge recommended). The page loads entirely from the file — no internet connection is needed after the first load, which fetches map tiles.

You will see:
- A **sidebar** on the left with a list of golf courses
- A **map** on the right showing all golf course polygons

---

## Understanding the Map

Polygons are color-coded:
- **Blue** — Verified courses (confirmed in both OpenStreetMap and Foursquare)
- **Red** — Unverified courses (found in only one source, or Foursquare-only placeholders shown as small hexagons)

Verified courses are **pre-checked** for export. OSM Only and Foursquare Only courses are **unchecked** by default — you decide which to include.

---

## Sidebar Controls

### Category Filters

The three checkboxes at the top of the sidebar control which categories are visible on the map **and** in the list below:

- **Verified (OSM + Foursquare)** — checked by default
- **OSM Only** — checked by default
- **Foursquare Only** — checked by default

Uncheck a category to temporarily hide those courses from both the map and the sidebar list. This is useful when you want to focus on reviewing one category at a time.

### Search

Type in the search bar to filter the list by name. The filter applies instantly as you type. Clear the text to show all courses again.

### Hover to Pan

When this checkbox is **checked** (the default), moving your mouse over a course name in the sidebar will automatically pan the map to that course. This is useful for quickly scanning through the list.

When **unchecked**, the map will not move on hover — you must click a course name to navigate to it. Use this mode when you want to browse the list without the map jumping around.

### The Course List

Each row in the list has:
- A **checkbox** on the left — this controls whether the course is included when you export
- A **color dot** matching the polygon color on the map
- The **course name** — click it to zoom to that course on the map

---

## Reviewing Courses

A suggested workflow for reviewing all courses:

1. **Turn on "Hover to Pan"** (checked by default)
2. **Uncheck the categories you're not reviewing yet** — for example, uncheck "Verified" and "Foursquare Only" to focus on "OSM Only" first
3. **Move your mouse down the list** — the map will pan to each course as you hover over its name
4. **For each course:**
   - If it looks correct and should be included: **check its checkbox**
   - If the polygon shape needs fixing: **right-click the polygon on the map** and choose **Edit Shape** (see "Editing a Polygon" below), then check its checkbox
   - If it's not a real golf course or should be excluded: **leave it unchecked**, or **right-click > Delete** to remove it entirely
5. **Repeat for the other categories**

---

## Editing a Polygon

To reshape a polygon (fix boundaries, expand a hexagon placeholder, etc.):

1. **Right-click** on the polygon on the map
2. A small menu appears — click **"Edit Shape"**
3. The polygon border changes to a **dashed line** and white vertex handles appear
4. A **yellow bar** appears at the top of the map confirming which course you're editing
5. **Drag any vertex** to move it
6. **Drag a midpoint handle** (the semi-transparent points between vertices) to add a new vertex
7. When you're done, stop editing by doing **any one** of these:
   - **Right-click the polygon again** and choose "Edit Shape" to toggle it off
   - Press the **Escape** key
   - Click the **"Done"** button in the yellow bar

Your changes are preserved — when you export, the updated shape will be saved.

---

## Deleting a Course

If a polygon does not represent a real golf course and should be permanently removed:

1. **Right-click** on the polygon on the map
2. Click **"Delete"** (highlighted in red)
3. The polygon is removed from both the map and the sidebar list

Deleted courses will not appear in exports. If you import the exported file later, the deletions will be preserved.

**Note:** There is no undo for delete within a session. If you delete something by mistake, you can reload the page to start fresh (you'll lose other unsaved work), or simply avoid exporting and reload.

---

## Exporting Your Work

When you're ready to save your progress:

1. Click the **"Export GeoJSON"** button at the bottom of the sidebar
2. A file named `nj_golf_courses_edited.geojson` will download to your computer
3. An alert will confirm how many courses were exported

**Only checked courses are included in the export.** Unchecked and deleted courses are excluded. Any polygon edits you made (dragged vertices, reshaped boundaries) are saved in the exported file.

The exported file also remembers which courses you deleted, so they won't reappear when you import later.

---

## Resuming Work Later (Import)

You do not need to finish everything in one session. To continue where you left off:

1. Open `nj_golf_courses_editable.html` in your browser (it always starts fresh)
2. Click the **"Import GeoJSON"** button at the bottom of the sidebar
3. Select your previously exported `.geojson` file
4. The map will restore your previous state:
   - Courses you had checked will be checked again
   - Polygon edits you made will be restored
   - Courses you deleted will be hidden again
   - Everything else resets to its default state
5. An alert will confirm how many courses were imported
6. Continue reviewing, editing, and checking
7. Export again when you're done (or when you want to save progress)

You can import and export as many times as you like. Each export is a complete snapshot of your current work.

---

## Tips

- **Use Google Satellite view** to verify polygon boundaries against real aerial imagery. Click the layer icon in the top-right corner of the map to switch between map styles.
- **Foursquare Only courses appear as small hexagons.** These are placeholder shapes centered on the Foursquare location. To include one, right-click it, choose "Edit Shape", and drag the vertices to trace the actual course boundary on the satellite view. Then check its checkbox.
- **The status bar** below the search box shows how many courses are visible and how many are checked for export — use it to track your progress.
- **Export often.** There is no auto-save. If your browser crashes or you accidentally close the tab, unsaved work is lost.
- **You can share the exported GeoJSON file** with others. They can import it into their own copy of the editable map to review or continue your work.

---

## Quick Reference

| Action | How |
|--------|-----|
| Show/hide a category | Check/uncheck the category checkbox at the top of the sidebar |
| Search for a course | Type in the search bar |
| Zoom to a course | Click its name in the sidebar |
| Pan to a course (quick scan) | Hover over its name (when "Hover to Pan" is on) |
| Mark a course for export | Check its checkbox in the sidebar |
| Edit a polygon's shape | Right-click the polygon on the map > "Edit Shape" |
| Finish editing | Right-click again, press Escape, or click "Done" |
| Delete a course | Right-click the polygon > "Delete" |
| Save your work | Click "Export GeoJSON" |
| Resume previous work | Click "Import GeoJSON" and select your saved file |
