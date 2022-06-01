#!/usr/bin/env python

import requests
#port = 55512
port = 55573
print(f'Performing post request on port {port}\n')
page='post'
#page='analyze'
key='kistlerfile'
#key='file'
#r = requests.post(f'http://deeplearning.ge.imati.cnr.it:{port}/{page}', data={key: 'fake base64-encoded data'})
r = requests.post(f'http://deeplearning.ge.imati.cnr.it:{port}/{page}', data={
										key:		'fake base64-encoded data',
										'username':	'extuser12',
										'password':	'extpassword1',
									})
'''
r = requests.post(f'http://deeplearning.ge.imati.cnr.it:{port}/{page}', data={
										key: 'fake base64-encoded data',
										'filename': 'fake filename',
									})
'''
# And done.
print(f'Answer:')
print(20*'-')
print(r.text) # displays the result body.
print(20*'-')

