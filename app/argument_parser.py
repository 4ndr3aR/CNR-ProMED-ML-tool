import sys
import argparse

def define_boolean_argument(parser, cmd_name, dst_variable, help_str):
	parser.add_argument('--'+cmd_name, dest=dst_variable, action='store_true', help=help_str)
	parser.add_argument('--no-'+cmd_name, dest=dst_variable, action='store_false')

def var2opt(dst_variable):
	return dst_variable.replace('_', '-'), dst_variable

