version     = "0.2.1"
author      = "bitstorm"
description = "Arrayfire wrapper for nim"
license     = "BSD"

# Dependencies

requires "nim >= 1.2.6"

when defined(nimdistros):
    import distros
    if detectOs(ArchLinux):
        foreignDep "arrayfire"


task tests, "Run all Arrayfire-Nim tests":
    exec "testament all"