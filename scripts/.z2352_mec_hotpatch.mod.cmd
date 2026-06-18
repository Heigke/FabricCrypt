savedcmd_z2352_mec_hotpatch.mod := printf '%s\n'   z2352_mec_hotpatch.o | awk '!x[$$0]++ { print("./"$$0) }' > z2352_mec_hotpatch.mod
