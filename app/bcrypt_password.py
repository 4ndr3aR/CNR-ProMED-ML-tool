#!/usr/bin/env python3

import os

import bcrypt

from colored import fg, bg, attr

rst = attr("reset")				# just to colorize text
user_pass_db = {}				# to be populated later in main, but needed to starlette_auth.BasicAuthBackend

def encrpyt_password(plaintext_password, salt_rounds=12, debug=False):
	salt = bcrypt.gensalt(salt_rounds)
	if debug:
		print(f'bcrypt generated salt: {salt}')
	hashed = bcrypt.hashpw(plaintext_password.encode('UTF8'), salt)
	if debug:
		print(f'bcrypt salted and hashed password: {hashed}')
	return hashed, salt

def verify_password(plaintext_password, hashed_password, salt=b'', debug=False):
	if debug:
		retrieved_salt = hashed_password.find(salt)
		print(f'Correct salt from bcrypt hashed password? {retrieved_salt}')
	hashed2 = bcrypt.hashpw(plaintext_password.encode('UTF8'), hashed_password)
	if debug:
		print(f'bcrypt salted and hashed password (to be verified): {hashed2}')
	match = (hashed_password == hashed2)
	if debug:
		print(f'The two salted and hashed passwords match? {match}')
	return hashed2, match

def verify_user_and_password(username, plaintext_password, debug=False):
	print(f'\n\nReceived username: {fg("chartreuse_2a")}{username}{rst} and password: {fg("turquoise_2")}{plaintext_password}{rst}\n\n')
	if username in user_pass_db:
		db_password = user_pass_db[username]
	else:
		msg = 'Authentication denied, no such user: '
		print(f'{fg("red_1")}{36*"-"}{rst}')
		print(f'{fg("red_1")}{msg}{rst}{fg("chartreuse_2a")}{username}{rst}')
		print(f'{fg("red_1")}{36*"-"}{rst}')
		print('\n')
		#return f"/post received username: {username} -> {msg}{username}"
		return False, msg

	hashed,  salt  = encrpyt_password(plaintext_password, debug=debug)
	hashed2, match = verify_password(plaintext_password=db_password, hashed_password=hashed, debug=debug)

	if match:
		print(f'{fg("chartreuse_2a")}{22*"-"}{rst}')
		print(f'{fg("chartreuse_2a")}Authentication granted{rst}')
		print(f'{fg("chartreuse_2a")}{22*"-"}{rst}')
		print('\n')
		return True, ''
	else:
		msg = 'Authentication denied, passwords do not match for: '
		print(f'{fg("red_1")}{50*"-"}{rst}')
		print(f'{fg("red_1")}{msg}{rst}{fg("chartreuse_2a")}{username}{rst}')
		print(f'{fg("red_1")}{50*"-"}{rst}')
		print('\n')
		return False, msg

def create_user_pass_db(debug=True):
	'''
	user_pass_db = {
				'user1': 'password1',
				'user2': 'password2',
				'user3': 'password3',
				'user4': 'password4',
			}
	'''
	external_user = os.environ['EXT_USERNAME']
	external_pass = os.environ['EXT_PASSWORD']
	user_pass_db[external_user] = external_pass
	print(f'Populated user_pass_db: {user_pass_db}')
	return user_pass_db


if __name__ == '__main__':

	debug=True

	hashed, salt = encrpyt_password('I love DigitBrain', debug=debug)
	verify_password(plaintext_password='I love DigitBrain', hashed_password=hashed, debug=debug)
