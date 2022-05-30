#!/usr/bin/env python3.8

import sys
import requests

if len(sys.argv) < 2:
	print(f'Please provide a KistlerFile name to read...')
	sys.exit()
if len(sys.argv) < 3:
	print(f'Please provide a port number...')
	sys.exit()
if len(sys.argv) >= 4:
	username = sys.argv[3]
	print(f'Using username: {username}')
if len(sys.argv) >= 5:
	password = sys.argv[4]
	print(f'Using password: {password}')

kfname = sys.argv[1]
print(f'Reading filename: {kfname}')

with open(kfname, 'rb') as kfd:
	kfdata = kfd.read()

print(f'{kfdata = }')

#port = 55563
port = int(sys.argv[2])
print(f'Performing post request on port {port}\n')

if len(sys.argv) < 5:
	r = requests.post(f'http://deeplearning.ge.imati.cnr.it:{port}/post', data={'filename': kfname, 'kistlerfile': kfdata})
else:
	r = requests.post(f'http://deeplearning.ge.imati.cnr.it:{port}/post', data={'filename': kfname, 'kistlerfile': kfdata, 'username': username, 'password': password})

# And done.
print(f'Answer:')
print(20*'-')
print(r.text) # displays the result body.
print(20*'-')

