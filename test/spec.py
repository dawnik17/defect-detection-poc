SPEC = {
    "cigarette_box": dict(
        item="a cigarette pack standing in a clear acrylic display holder",
        labels=dict(
            good="fully closed: the hinged lid sits flush on the body with straight, even top edges; "
                 "no inner foil, cardboard or cigarettes visible",
            opened="the lid is lifted, folded back, torn off or even slightly ajar (a gap or step between "
                   "lid and body), exposing the inner foil, the cardboard inner frame or the cigarettes"),
        shots=[
            ("good", "train/good/080_001.jpg",
             "The lid sits flush on the body with straight, even top edges, and nothing inside the pack is visible."),
            ("opened", "test/opened/041_004.jpg",
             "The lid is folded back and the bare cardboard inner frame sticks up above the top of the pack."),
        ]),
    "drink_bottle": dict(
        item="a plastic or glass drink bottle on a shelf",
        labels=dict(
            good="cap screwed fully down with no gap above its tamper ring; label smooth and complete",
            cap_half_open="cap present but loosened or raised: a visible gap between the cap and its tamper "
                          "ring, or neck threads showing below the cap",
            cap_open="no cap: the bare threaded neck and open mouth are visible, often with only the tamper ring left",
            surface_damage="label torn, peeled, creased or hanging loose, or the bottle surface scratched"),
        shots=[
            ("good", "train/good/073_001.jpg",
             "The light-blue cap is screwed fully down with no gap above the neck ring, and the label is smooth and complete."),
            ("cap_half_open", "test/cap_half_open/073_012.jpg",
             "The light-blue cap sits raised above its tamper ring, leaving a gap where the neck threads show."),
            ("cap_open", "test/cap_open/008_011.jpg",
             "There is no cap: the bare threaded neck and open mouth are visible above the leftover orange tamper ring."),
            ("surface_damage", "test/surface_damage/007_012.jpg",
             "The blue label is torn and a long strip of it hangs loose from the bottle."),
        ]),
    "drink_can": dict(
        item="a drink can or a carton drink box on a shelf",
        labels=dict(
            good="smooth body with straight edges and intact print; a carton sold with a straw still has its "
                 "clear-wrapped straw attached to a panel",
            deformation="dented, crushed, creased or buckled can wall or carton panel, edge or corner",
            straw_missing="a carton whose attached straw is gone: the panel where the wrapped straw is normally "
                          "glued is bare",
            surface_damage="scratches, scuffs or scraped-off print or paint on the can or carton, often white or "
                           "silver marks"),
        shots=[
            ("good", "train/good/024_014.jpg",
             "The can body is smooth and round, with intact print and no dents or scratches."),
            ("good", "train/good/042_001.jpg",
             "The carton has straight edges and intact print, and its clear-wrapped straw is attached diagonally across the panel."),
            ("deformation", "test/deformation/049_005.jpg",
             "The top of the carton is crushed: the upper edge is bent down and creased at the corner."),
            ("straw_missing", "test/straw_missing/042_005.jpg",
             "The carton is intact, but the panel that should carry the clear-wrapped straw is bare: the straw is missing."),
            ("surface_damage", "test/surface_damage/008_004.jpg",
             "A patch of the brown print on the can is scraped off, leaving white scratch marks."),
        ]),
    "food_bottle": dict(
        item="a cylindrical food container such as a cup-noodle cup, chip canister or jar on a shelf",
        labels=dict(
            good="round, undented body and rim; lid or foil seal flat and fully attached; wrapper intact",
            deformation="dented, crushed or buckled container wall or rim",
            opened="lid or foil/paper seal peeled back, torn open or missing at the top",
            surface_damage="scratches, scuffs, punctures, small tears or scraped-off print on the wrapper"),
        shots=[
            ("good", "train/good/032_005.jpg",
             "The cup is round and undented, its lid is flat and fully attached at the rim, and the wrapper print is intact."),
            ("deformation", "test/deformation/036_019.jpg",
             "The rim of the cup is crushed inward and the lid edge above it is crumpled."),
            ("opened", "test/opened/033_028.jpg",
             "The foil lid is peeled back from the rim and sticks up, leaving the cup open."),
            ("surface_damage", "test/surface_damage/006_029.jpg",
             "A patch of the red printed wrapper is scraped off, leaving white scuff marks."),
        ]),
    "food_box": dict(
        item="a rectangular cardboard food box on a shelf",
        labels=dict(
            good="flat faces, straight edges and square corners; all flaps closed; print intact",
            deformation="crushed, dented, bent or twisted box: caved-in faces, crumpled corners or warped edges",
            opened="a flap open or lifted, or the tear strip or perforation torn, leaving an open top or a ragged edge",
            surface_damage="small scuffs, scratches or tears where the print is scraped off down to white cardboard"),
        shots=[
            ("good", "train/good/034_001.jpg",
             "The box has flat faces, straight edges and closed flaps, and the print is intact."),
            ("deformation", "test/deformation/038_020.jpg",
             "The box is crushed and twisted along its length, with the side panel buckled inward."),
            ("opened", "test/opened/024_013.jpg",
             "The top flaps are open and folded up, exposing the inside of the box."),
            ("surface_damage", "test/surface_damage/049_013.jpg",
             "White scratch streaks run across the blue front panel where the print has been scraped off."),
        ]),
    "food_package": dict(
        item="a flexible food bag or pouch on a shelf",
        labels=dict(
            good="film intact and seals closed; normal wrinkles and handling creases are fine",
            broken="bag torn, ripped, punctured or cut open: a hole, a ragged tear or an open seal",
            surface_anomaly="foreign material stuck on the bag that is not part of the printed design, such as a "
                            "scrap of plastic, foil, paper or tape, or a stain"),
        shots=[
            ("good", "train/good/033_005.jpg",
             "The bag is sealed and its film is intact; the wrinkles are normal handling creases."),
            ("broken", "test/broken/086_011.jpg",
             "The top of the bag is ripped open with ragged edges, exposing the silver inner lining."),
            ("surface_anomaly", "test/surface_anomaly/063_013.jpg",
             "A small scrap of orange plastic is stuck on the front of the bag; it is not part of the printed design."),
        ]),
}
CATS = tuple(SPEC)

# Prompt
def prompt(cat):
    s = SPEC[cat]
    defs = "\n".join(f"- {k}: {v}" for k, v in s["labels"].items())
    return (f"You are a retail quality inspector. The image shows {s['item']}.\nLabels:\n{defs}\n"
            "Check the whole item: top, cap or lid, edges, corners, seams and every visible face. "
            "Ignore the shelf, background and display holder.\n"
            f"Reply with one sentence of visual evidence, then a final line: Answer: <{'|'.join(s['labels'])}>")
