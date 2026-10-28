* Added an optional WithBall mesh/SDF collision profile to reduce passive drift
  from primitive-mesh contacts, while preserving the rendered sphere, materials
  and analytic mass properties. Existing task defaults and replay were retained;
  the proxy used an explicit 50 micrometer radial approximation budget.
