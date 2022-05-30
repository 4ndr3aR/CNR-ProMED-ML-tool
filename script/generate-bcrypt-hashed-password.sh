#!/bin/bash

password="$1"

htpasswd -bnBC 12 "" "$password" | tr -d ':\n'

exit 0
