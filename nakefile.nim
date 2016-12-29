import nake
import os

const testDir="test"
const targetDir="target"


let skip_tests :seq[string] = @[]


task defaultTask, "Run Tests":
  if not existsDir(targetDir):
    createDir(targetDir)

  for nakeFile in walkFiles testDir & "/*_test.nim":    
    var (_, name, _) = splitFile(nakeFile)
    if name in skip_tests:
      continue
    echo "running test $1"%name
    direShell(nimExe,"cpp","-r","--lineDir:on","--verbosity:0","--colors:on","--hints:off",
      "--debugger:native","--debuginfo","--out=$1$2$3"%[targetDir,$DirSep,name],nakeFile)


task "transform","Transform header":
  direShell(nimExe,"c","-r","--lineDir:on","--verbosity:0","--colors:on","--hints:off",
      "--debuginfo","--out=target/process scripts/process.nim")

task "doc", "generate documentation":
  direShell(nimExe,"doc2","--docSeeSrcUrl:txt","--out=docs/ArrayFireNim.html","ArrayFireNim.nim")
