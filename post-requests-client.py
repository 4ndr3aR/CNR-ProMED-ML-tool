#!/usr/bin/env python

import sys
import requests

model_classes = { 
                        'bar'                      : [], 
                        'bridge'                   : [], 
                        'front_lower_bridge'       : [], 
                        'solo_crown'               : [], 
                        'upper_subperiost_implant' : [], 
                   }

classes = [k for k,v in model_classes.items()]

if len(sys.argv) > 1:
	clas = sys.argv[1]
else:
	from random import randrange
	clas = classes[randrange(5)]

if len(sys.argv) > 2:
	host = sys.argv[2]
else:
	host = 'deeplearning.ge.imati.cnr.it'

if len(sys.argv) > 3:
	port = sys.argv[3]
else:
	port = 55573

print(f'Performing post request on {host}:{port} sending class: {clas}\n')
page='post'
key='manufacturing-class'
r = requests.post(f'http://{host}:{port}/{page}', data={
										key:		clas,		# the class for which to predict values
										'username':	'extuser1',
										'password':	'extpassword1',
									})
# And done.
print(f'Answer:')
print(20*'-')
print(r.text) # displays the result body.
print(20*'-')

