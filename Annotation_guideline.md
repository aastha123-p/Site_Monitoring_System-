# 🏷️ Construction Site Dataset Annotation Guidelines

This manual defines the official labeling and annotation standards for the **AI Construction Site Safety & Hazard Monitoring System**. All annotators, machine learning engineers, and data contributors must strictly adhere to these rules to ensure high annotation consistency, minimize class confusion, and maximize YOLOv8 detection accuracy.

---

## 📌 Table of Contents

- [1. Data Format & Coordinate Standards](#1-data-format--coordinate-standards)
- [2. Class Taxonomy & Definitions (10 Classes)](#2-class-taxonomy--definitions-10-classes)
- [3. Hierarchical & Nested PPE Labeling Standard](#3-hierarchical--nested-ppe-labeling-standard)
- [4. Bounding Box Tightness & Boundary Rules](#4-bounding-box-tightness--boundary-rules)
- [5. Handling Occlusion, Truncation & Crowd Scenes](#5-handling-occlusion-truncation--crowd-scenes)
- [6. Minimum Size & Distance Guidelines](#6-minimum-size--distance-guidelines)
- [7. Edge Case Resolution & Gray-Area Matrix](#7-edge-case-resolution--gray-area-matrix)
- [8. Quality Assurance & Automated Validation](#8-quality-assurance--automated-validation)

---

## 1. Data Format & Coordinate Standards

### YOLO Bounding Box Specification
Every labeled image must have a corresponding `.txt` file with the exact same basename:
```text
images/frame_00124.jpg ➔ labels/frame_00124.txt
```

Each line in the `.txt` file represents exactly one labeled bounding box in standard normalized format:
```text
<class_id> <x_center> <y_center> <width> <height>
```

| Field | Range | Description |
| :--- | :---: | :--- |
| `class_id` | `0` – `9` | Integer ID corresponding to the target class |
| `x_center` | `[0.0, 1.0]` | Normalized horizontal center coordinate ($X / W_{\text{image}}$) |
| `y_center` | `[0.0, 1.0]` | Normalized vertical center coordinate ($Y / H_{\text{image}}$) |
| `width` | `[0.0, 1.0]` | Normalized width of the bounding box ($W_{\text{box}} / W_{\text{image}}$) |
| `height` | `[0.0, 1.0]` | Normalized height of the bounding box ($H_{\text{box}} / H_{\text{image}}$) |

> ⚠️ **Strict Constraint**: All normalized values must remain strictly within `[0.000000, 1.000000]`. Box boundaries must never extend outside the image dimensions.

---

## 2. Class Taxonomy & Definitions (10 Classes)

The dataset defines **10 target classes** spanning personnel, personal protective equipment (PPE), heavy machinery, and safety markers.

```yaml
names:
  0: person
  1: hardhat
  2: no_hardhat
  3: safety_vest
  4: no_safety_vest
  5: machinery
  6: crane
  7: excavator
  8: truck
  9: safety_cone
```

---

### Class 0: `person`
* **Definition**: Any human on the construction site regardless of role (construction worker, engineer, surveyor, safety inspector, machine operator, visitor, or pedestrian).
* **Scope**: Bounding box must enclose the **entire visible body** from head (or hardhat) down to the feet (or boots).
* **If Partial / Occluded**: If upper torso and head are visible, annotate the full visible extent.
* **Do NOT Label**:
  * Statues, billboards, mannequins, or drawings.
  * Reflections of humans in vehicle mirrors or glass.

---

### Class 1: `hardhat`
* **Definition**: Any protective safety helmet or hardhat designed for construction head protection.
* **Colors**: Any industrial safety color (white, yellow, orange, blue, red, green, brown).
* **Scope**: Tightly bound around the helmet shell, including visor/brim and mounted accessories (headlamps, face shields, ear defenders).
* **Include**:
  * Hardhats actively worn on a worker's head.
  * Hardhats placed on tables, hung on machinery, or carried in hand (if clearly distinguishable).

---

### Class 2: `no_hardhat`
* **Definition**: A human head that is **not** protected by an approved safety hardhat inside an active site zone.
* **Includes**:
  * Bare head with exposed hair or bald head.
  * Baseball caps, beanies, sunhats, turbans, headscarves, or hoodies without a hardhat on top.
* **Scope**: Bounding box should tightly cover the **head and hair region** (from chin to top of head).
* **Do NOT Label**:
  * Do not label `no_hardhat` if the worker is wearing a compliant hardhat.

---

### Class 3: `safety_vest`
* **Definition**: High-visibility (Hi-Vis) safety vests, coats, or reflective harnesses worn over clothing.
* **Visual Identifiers**: Fluorescent colors (neon lime-yellow, fluorescent orange, bright red) **and/or** silver/grey retroreflective stripes.
* **Scope**: Bound the entire visible area of the vest/jacket covering the torso and arms (if sleeved hi-vis jacket).
* **Include**:
  * Open (unzipped) or closed vests.
  * Fall protection harnesses with reflective webbing.

---

### Class 4: `no_safety_vest`
* **Definition**: A worker whose upper torso is **not** covered by high-visibility safety gear.
* **Visual Identifiers**: Standard street clothing, dark t-shirts, flannel shirts, denim jackets, ordinary hoodies, or bare torso.
* **Scope**: Bound the **torso region** (shoulders down to the waistline/belt).
* **Do NOT Label**:
  * Do not label if the worker is wearing any approved high-vis safety vest or jacket.

---

### Class 5: `machinery`
* **Definition**: Stationary or non-vehicular construction equipment and motorized site tools not covered by classes 6, 7, or 8.
* **Includes**:
  * Generators, air compressors, industrial light towers.
  * Stationary concrete mixers, plate compactors, road rollers.
  * Piling rigs, scissor lifts, boom lifts, hydraulic pumps.
* **Do NOT Include**:
  * Handheld power tools (drills, angle grinders).
  * Cranes, excavators, or haul trucks (use specific classes 6, 7, 8).

---

### Class 6: `crane`
* **Definition**: Heavy lifting machines designed to raise, lower, and transport materials horizontally and vertically.
* **Includes**:
  * **Tower Cranes**: Mast, jib, counter-jib, and operator cab.
  * **Mobile / Crawler Cranes**: Tracked or wheeled chassis with telescopic or lattice booms.
  * **Truck-Mounted Cranes (Boom Trucks)**: If the crane boom is deployed and is the dominant function.
* **Scope**: Bound the entire structure including boom, chassis, and cabin (excluding thin cable/hook lines extending far from the machine).

---

### Class 7: `excavator`
* **Definition**: Heavy digging and earthmoving machinery equipped with a hydraulic boom, stick (dipper arm), and bucket.
* **Includes**:
  * Tracked and wheeled hydraulic excavators.
  * Mini-excavators and micro-diggers.
  * Backhoe loaders (excavator arm visible).
* **Scope**: Bound the vehicle body (tracks/wheels, cabin, counterweight) **together with** the extended hydraulic arm and bucket.

---

### Class 8: `truck`
* **Definition**: Heavy-duty motorized transport vehicles operating on or entering the construction site.
* **Includes**:
  * Articulated dump trucks, rigid haul trucks.
  * Transit concrete mixer trucks.
  * Flatbed delivery haulers, low-bed equipment trailers.
  * Heavy site pickup trucks (e.g. Ford F-350 / RAM 3500 work trucks).
* **Do NOT Include**:
  * Ordinary personal passenger sedans parked outside the site perimeter.

---

### Class 9: `safety_cone`
* **Definition**: Portable, conical safety markers used to redirect traffic, cordon off danger zones, or highlight tripping hazards.
* **Includes**:
  * Standard orange traffic cones (with or without reflective collars).
  * High-visibility delineator posts and hazard bollards.
* **Scope**: Bounding box from the flat base up to the top tip.

---

## 3. Hierarchical & Nested PPE Labeling Standard

To enable multi-task safety compliance, **every worker must receive nested, overlapping annotations**. 

### The Golden Rule of PPE Labeling
When annotating a person, you must create **three distinct bounding boxes**:
1. **Outer Box**: `person` (Class `0`) covering the full visible body.
2. **Head Box**: Either `hardhat` (Class `1`) **OR** `no_hardhat` (Class `2`).
3. **Torso Box**: Either `safety_vest` (Class `3`) **OR** `no_safety_vest` (Class `4`).

```text
 ┌──────────────────────────────────────────────┐
 │ [Class 0: person]                            │
 │                                              │
 │       ┌──────────────────────────────┐       │
 │       │ [Class 1: hardhat]           │       │
 │       │  (or Class 2: no_hardhat)    │       │
 │       └──────────────────────────────┘       │
 │                       │                      │
 │       ┌──────────────────────────────┐       │
 │       │ [Class 3: safety_vest]       │       │
 │       │  (or Class 4: no_safety_vest)│       │
 │       │                              │       │
 │       └──────────────────────────────┘       │
 │                       │                      │
 │                     Legs                     │
 │                                              │
 └──────────────────────────────────────────────┘
```

> ❌ **CRITICAL ERROR**: Never omit the `person` box just because you labeled the `hardhat` and `vest`. The detector must learn both the human silhouette and the individual protective gear.

---

## 4. Bounding Box Tightness & Boundary Rules

### 1. Tightness Standard
* The bounding box must touch the **outermost visible pixels** of the target object.
* Allow a maximum margin of **1 to 2 pixels** of background context.
* Do **not** leave loose borders—excessive background inside a bounding box damages localization accuracy (`mAP@0.50:0.95`).

```text
       WRONG (Too Loose)                   CORRECT (Tight Fit)
 ┌───────────────────────────┐         ┌─────────────────────┐
 │                           │         │   /\___/\           │
 │       /\___/\             │         │  ( o   o )          │
 │      ( o   o )            │         │  (  =^=  )          │
 │      (  =^=  )            │         │   (_____)           │
 │       (_____)             │         └─────────────────────┘
 │                           │
 └───────────────────────────┘
```

### 2. Irregular Shapes (Excavators & Cranes)
* For articulated machines (such as an excavator with an arm extended upward), draw the box around the **complete bounding rectangle** containing the chassis, cabin, arm, and bucket.
* Do not chop off the bucket or arm just because it extends high into the sky.

---

## 5. Handling Occlusion, Truncation & Crowd Scenes

### 1. The 20% Visibility Rule
* **Visibility $\ge 20\%$**: Annotate the object. Draw the box tightly around the **visible** portion of the object.
* **Visibility $< 20\%$**: Do **not** annotate. If an object is so heavily obstructed that a human cannot identify it with certainty without context, skip it.

### 2. Partial Occlusion by Site Scaffolding / Poles
* When an object is split by thin scaffolding bars, wires, or handrails:
  * Draw **one single bounding box** encompassing the whole object across the thin obstruction.
  * Do **not** split the person into two separate boxes because of a handrail.

### 3. Image Border Truncation
* When a worker or vehicle is cut off by the camera boundary:
  * Draw the box tightly up to the edge of the image frame (`0.0` or `1.0`).
  * Never extrapolate invisible body parts outside the image borders.

### 4. Overlapping & Crowd Scenes
* In group meetings, toolboxes, or congested areas:
  * Each individual must have their own separate `person` box.
  * Each visible head and torso must have their own respective PPE boxes.
  * Do not merge two people into a single box.

---

## 6. Minimum Size & Distance Guidelines

To prevent label noise from heavily pixelated distant objects:

| Object Type | Minimum Bounding Box Dimension | Action if Below Threshold |
| :--- | :---: | :--- |
| **`person`** | $20 \times 20$ pixels | Skip if features are unidentifiable |
| **`hardhat` / `no_hardhat`** | $10 \times 10$ pixels | Skip if head area is a blurred smudge |
| **`safety_vest` / `no_safety_vest`** | $15 \times 15$ pixels | Skip if torso color cannot be determined |
| **`safety_cone`** | $10 \times 10$ pixels | Skip if just 2–3 blurred pixels |
| **`machinery` / `crane` / `truck`** | $30 \times 30$ pixels | Skip if unrecognizable in the far background |

---

## 7. Edge Case Resolution & Gray-Area Matrix

| Scenario | Correct Classification | Explanation |
| :--- | :---: | :--- |
| **Worker wearing a baseball cap or beanie** | `no_hardhat` | Baseball caps and beanies offer zero impact protection and are safety violations. |
| **Worker wearing a hoodie pulled up over head** | `no_hardhat` | Unless an approved hardhat is visibly mounted over the hood, it is non-compliant. |
| **Worker wearing hardhat over a winter liner / bandana** | `hardhat` | If the rigid shell of the hardhat is on top, it is classified as `hardhat`. |
| **Hardhat resting on the ground or carried in hand** | `hardhat` | Annotate the hardhat object, but annotate the worker's head as `no_hardhat`. |
| **Worker wearing a bright yellow / orange plain t-shirt** | `no_safety_vest` | Ordinary bright sportswear lacks certified retroreflective strips and does not meet Hi-Vis standards. |
| **Worker wearing Hi-Vis jacket with reflective stripes** | `safety_vest` | Full-sleeve reflective jackets qualify as `safety_vest`. |
| **Worker wearing safety harness with reflective bands** | `safety_vest` | Fall-arrest harness with reflective webbing qualifies as PPE torso coverage. |
| **Machine operator visible inside glass cabin** | `person` (+ PPE if clear) | If the operator is clearly visible through the glass, annotate. If heavily tinted or obscured by glare, skip. |
| **Pickup truck on site** | `truck` | Commercial pickup trucks (contractor work trucks) on site qualify as `truck`. Passenger sedans do not. |
| **Forklift / Scissor Lift** | `machinery` | Industrial lifts belong in `machinery` (not truck or crane). |
| **Cement Mixer Truck** | `truck` | Transit mixers on a vehicular chassis belong in `truck`. Portable stationary mixers belong in `machinery`. |

---

## 8. Quality Assurance & Automated Validation

Before committing new annotations to the dataset, run the automated verification scripts provided in `scripts/dataset_preparation/`:

### Step 1: Check for Coordinate & Syntax Errors
```bash
python scripts/dataset_preparation/validate_dataset.py --images data/raw/images --labels data/raw/labels
```
* Verifies all boxes are normalized within `[0.0, 1.0]`.
* Confirms all class IDs are integers between `0` and `9`.
* Flags empty label files or missing pairs.

### Step 2: Audit Dataset Dimensions & Zero-Area Boxes
```bash
python scripts/dataset_preparation/audit_dataset.py
```
* Detects unreadable image formats and zero-width/height boxes.

### Step 3: Check Split Leakage Before Merging
```bash
python scripts/dataset_preparation/check_split_leakage.py
```
* Ensures that near-duplicate frames from the same camera feed have not leaked across train, validation, and test partitions.

---

### Annotator Checklist (Quick Reference)

- [ ] Every person has an outer `person` box.
- [ ] Every person has a corresponding `hardhat` OR `no_hardhat` box.
- [ ] Every person has a corresponding `safety_vest` OR `no_safety_vest` box.
- [ ] Bounding boxes fit tightly around the visible boundaries (no sloppy margins).
- [ ] Partially visible objects follow the 20% visibility rule.
- [ ] Plain bright clothing is correctly distinguished from certified reflective vests.
- [ ] Automated validation script passes with zero errors.
