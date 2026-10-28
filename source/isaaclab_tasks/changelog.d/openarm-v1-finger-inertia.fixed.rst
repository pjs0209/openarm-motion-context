* Corrected recognized legacy OpenArm V1 finger inertias on the Newton import path using
  upstream ``enactic/openarm_description`` commit ``62ec8259b1513d9212d0dacc7c496f52b62be30f``.
  Replaced both principal moments and their axes while preserving mass, COM, geometry,
  source USD layers and the PhysX path. Left other mechanical models to normal validation.
