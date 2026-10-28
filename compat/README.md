# Research compatibility sources

`skrl_1_4_3/` contains the pinned Python source and distribution metadata used by
the OpenArm MAPPO checkpoint path. The upstream MIT license is preserved at
`skrl_1_4_3/skrl-1.4.3.dist-info/licenses/LICENSE`; its terms apply to that source,
not this repository's root license. Bytecode, install records and caches are not
included. Other skrl algorithms are retained to preserve the original distribution.

The Newton SDF backport script is separate and checks the installed Newton version
and original source hash before changing its two synchronization sites. Read
`NEWTON_SDF_BACKPORT.md` before applying it. The site-package file itself is not
vendored or silently modified by the research-overlay installer.
