# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: options.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='options.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\roptions.proto\"f\n\x19GestureRecognizierOptions\x12\x12\n\nbroker_uri\x18\x01 \x01(\t\x12\x12\n\nzipkin_uri\x18\x02 \x01(\t\x12\x11\n\tgroup_ids\x18\x03 \x03(\r\x12\x0e\n\x06period\x18\x04 \x01(\x01\x62\x06proto3')
)




_GESTURERECOGNIZIEROPTIONS = _descriptor.Descriptor(
  name='GestureRecognizierOptions',
  full_name='GestureRecognizierOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='broker_uri', full_name='GestureRecognizierOptions.broker_uri', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='zipkin_uri', full_name='GestureRecognizierOptions.zipkin_uri', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='group_ids', full_name='GestureRecognizierOptions.group_ids', index=2,
      number=3, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='period', full_name='GestureRecognizierOptions.period', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17,
  serialized_end=119,
)

DESCRIPTOR.message_types_by_name['GestureRecognizierOptions'] = _GESTURERECOGNIZIEROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GestureRecognizierOptions = _reflection.GeneratedProtocolMessageType('GestureRecognizierOptions', (_message.Message,), dict(
  DESCRIPTOR = _GESTURERECOGNIZIEROPTIONS,
  __module__ = 'options_pb2'
  # @@protoc_insertion_point(class_scope:GestureRecognizierOptions)
  ))
_sym_db.RegisterMessage(GestureRecognizierOptions)


# @@protoc_insertion_point(module_scope)