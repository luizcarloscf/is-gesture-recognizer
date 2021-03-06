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
  serialized_pb=_b('\n\roptions.proto\"\x8f\x01\n\x19GestureRecognizierOptions\x12\x12\n\nbroker_uri\x18\x01 \x01(\t\x12\x12\n\nzipkin_uri\x18\x02 \x01(\t\x12\x11\n\tgroup_ids\x18\x03 \x03(\r\x12\x11\n\texec_time\x18\x04 \x01(\x01\x12$\n\tskeletons\x18\x05 \x01(\x0b\x32\x11.SkeletonsOptions\"K\n\x10SkeletonsOptions\x12\x15\n\rmin_keypoints\x18\x01 \x01(\r\x12\x0f\n\x07x_range\x18\x02 \x03(\x01\x12\x0f\n\x07y_range\x18\x03 \x03(\x01\x62\x06proto3')
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
      name='exec_time', full_name='GestureRecognizierOptions.exec_time', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='skeletons', full_name='GestureRecognizierOptions.skeletons', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=18,
  serialized_end=161,
)


_SKELETONSOPTIONS = _descriptor.Descriptor(
  name='SkeletonsOptions',
  full_name='SkeletonsOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_keypoints', full_name='SkeletonsOptions.min_keypoints', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='x_range', full_name='SkeletonsOptions.x_range', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y_range', full_name='SkeletonsOptions.y_range', index=2,
      number=3, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=163,
  serialized_end=238,
)

_GESTURERECOGNIZIEROPTIONS.fields_by_name['skeletons'].message_type = _SKELETONSOPTIONS
DESCRIPTOR.message_types_by_name['GestureRecognizierOptions'] = _GESTURERECOGNIZIEROPTIONS
DESCRIPTOR.message_types_by_name['SkeletonsOptions'] = _SKELETONSOPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GestureRecognizierOptions = _reflection.GeneratedProtocolMessageType('GestureRecognizierOptions', (_message.Message,), dict(
  DESCRIPTOR = _GESTURERECOGNIZIEROPTIONS,
  __module__ = 'options_pb2'
  # @@protoc_insertion_point(class_scope:GestureRecognizierOptions)
  ))
_sym_db.RegisterMessage(GestureRecognizierOptions)

SkeletonsOptions = _reflection.GeneratedProtocolMessageType('SkeletonsOptions', (_message.Message,), dict(
  DESCRIPTOR = _SKELETONSOPTIONS,
  __module__ = 'options_pb2'
  # @@protoc_insertion_point(class_scope:SkeletonsOptions)
  ))
_sym_db.RegisterMessage(SkeletonsOptions)


# @@protoc_insertion_point(module_scope)
