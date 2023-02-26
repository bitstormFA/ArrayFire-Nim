when not compiles(nimVersion):
  const nimVersion = (major: NimMajor, minor: NimMinor, patch: NimPatch)

when nimVersion >= (1, 3, 3):
  switch("backend", "cpp")
switch("path", "$projectDir/../src")
switch("outdir", "$projectDir/../bin")