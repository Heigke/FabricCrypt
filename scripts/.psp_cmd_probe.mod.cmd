savedcmd_psp_cmd_probe.mod := printf '%s\n'   psp_cmd_probe.o | awk '!x[$$0]++ { print("./"$$0) }' > psp_cmd_probe.mod
