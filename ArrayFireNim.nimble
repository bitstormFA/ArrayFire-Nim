version     = "0.2.2"
author      = "bitstorm"
description = "Arrayfire wrapper for nim"
license     = "BSD"

# Dependencies

requires "nim >= 1.2.0"

when defined(nimdistros):
    import distros
    if detectOs(ArchLinux):
        foreignDep "arrayfire"


task tests, "Run all Arrayfire-Nim tests":
    exec "testament all"
