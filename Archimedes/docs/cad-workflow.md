# Modelling in the Workbench: CAD-style interaction

The Geometry stage works the way SolidWorks and CATIA users expect: pick faces, drag them, edit dimensions on the
model, edit features with OK / Cancel, and navigate with the usual mouse buttons. Every model edit can be undone.

## Feature tree (left)

| Row | Is | Click | Double-click | Right-click |
|---|---|---|---|---|
| Origin · planes | reference planes | shows / hides them | | |
| **Sketch1 — L-profile** | width, height, wall | highlights its faces | PropertyManager | edit / menu |
| **Boss-Extrude1** | depth (blind) | highlights front and back | PropertyManager | edit / menu |
| **Fillet1** | re-entrant corner radius | highlights the fillet | PropertyManager | edit, **suppress / unsuppress** |

The **PropertyManager** replaces the right panel while you edit a feature: changes preview live, **✓ OK** (Enter)
keeps them, **✕ Cancel** (Esc) puts everything back. The same ✓ / ✕ sits in the viewport's confirmation corner
(top right). A whole edit session undoes as one step.

## On the model

| Do | Result |
|---|---|
| hover a face | pre-highlight (light blue) |
| click a face | select it (blue), its feature lights up in the tree, a context toolbar offers *Normal to · edit dimension · edit feature* |
| drag the face's **arrow** (Instant3D) | resize live: the dimension that drives the face follows the mouse with a ruler, snapping to 5 mm (wall and fillet: 1 mm; hold **Shift** for 1 mm) |
| click a **dimension** label, or double-click a face | the **Modify** box: type a value (live preview), mouse wheel / ↑↓ spin it, Enter ✓, Esc ✕ |
| right-click | context menu: edit dimension / feature, suppress, Normal to, zoom to fit, previous view, view orientation, display style, section view, planes, undo / redo, mouse style |

Which face drives which dimension:

| Face | Dimension |
|---|---|
| flange end | Width (D1@Sketch1) |
| top | Height (D2@Sketch1) |
| web inner face, flange underside | Wall (D3@Sketch1) |
| front, back | Depth (D1@Boss-Extrude1) |
| fillet | Radius (D1@Fillet1) |
| web outer face, base | none (the base carries the fixed support) |

The view stays put while you drag, then eases onto the resized part. The status line shows the selection and how
long the rebuild took.

## Navigation

| | SolidWorks (default) | CATIA (View menu or right-click) |
|---|---|---|
| rotate | left drag or middle drag | middle + left / right drag |
| pan | Ctrl + middle drag, or right drag | middle drag |
| zoom | wheel (at the cursor), Shift + middle drag | wheel (at the cursor), Ctrl + middle drag |
| zoom to fit | **F**, or double-click the middle button | same |

Left drag and right drag work on any mouse or trackpad in both styles.

| Key | Does |
|---|---|
| **Space** | view orientation: Front, Back, Left, Right, Top, Bottom, Isometric, Normal to |
| Ctrl+1 … Ctrl+7 | the same standard views (if the browser doesn't keep the shortcut for tabs) |
| Ctrl+8 | Normal to the selected face |
| ←→↑↓ | rotate 15° (Shift: 90°); Ctrl + arrows pan |
| Ctrl+Shift+Z | previous view |
| Ctrl+Z / Ctrl+Y | undo / redo model edits (dimensions, features, suppress, material, boundary conditions) |
| Home | reset the camera |
| Esc | cancel the Modify box or the feature edit, then clear the selection |

The heads-up toolbar at the top of the viewport has the same view tools: fit, previous view, view orientation,
Normal to, section view, display style (shaded with edges / wireframe / nodes), planes, undo and redo.
