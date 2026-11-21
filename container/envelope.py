"""The Workspace's wire payload: one native string-key mapping as MessagePack inside a WorkspaceEnvelope, and its bounds.

It depends only on msgpack. A timezone-aware datetime crosses as a MessagePack timestamp and arrives as a UTC datetime.
An exact Decimal crosses as a registered MessagePack extension holding its exact scientific string and arrives as the
same Decimal: sign, coefficient digits and exponent are kept. A caller frames its payloads the same way.
"""

from decimal import Decimal, InvalidOperation

from msgpack import ExtType, packb, unpackb

# A Workspace request or answer has no size limit of its own, so each message is bound only by protobuf's: 2 GB less one
# byte, which is also the largest gRPC message setting. The protobuf framing is outside the native payload, so the
# payload bound is the rest.
PROTOBUF_MESSAGE_LIMIT_BYTES = 2 * 1024 * 1024 * 1024 - 1
ENVELOPE_FRAME_BYTES = 16
PAYLOAD_LIMIT_BYTES = PROTOBUF_MESSAGE_LIMIT_BYTES - ENVELOPE_FRAME_BYTES
# A file read in parts asks for at most this many bytes at a time. The Workspace holds a part's bytes, its MessagePack
# payload and its protobuf message at once, so a part keeps a read well inside the smallest Workspace memory setting
# while a file of any size is read whole in parts.
READ_PART_LIMIT_BYTES = 32 * 1024 * 1024
# The registered MessagePack extension code of an exact Decimal.
DECIMAL_EXTENSION_CODE = 1


def encode_extension(value):
    """The MessagePack extension of a value with no native MessagePack form; only an exact Decimal has one."""
    if type(value) is not Decimal:
        raise TypeError(f"{type(value).__name__} has no service envelope encoding")
    extension = ExtType(DECIMAL_EXTENSION_CODE, str(value).encode("ascii"))
    return extension


def decode_extension(code: int, data: bytes):
    """The native value of one registered MessagePack extension; an unregistered or malformed one raises ValueError."""
    if code != DECIMAL_EXTENSION_CODE:
        raise ValueError(f"service envelope extension {code} is not registered")
    try:
        value = Decimal(data.decode("ascii"))
    except (UnicodeDecodeError, InvalidOperation) as err:
        raise ValueError("service envelope Decimal extension is malformed") from err
    return value


def encode_envelope(value: dict, limit: int, owner: str) -> bytes:
    """Encode one native mapping for `owner`'s envelope.

    Raises ValueError when the value is not a dict or its encoding exceeds `limit` bytes; TypeError for a value the
    envelope cannot encode.
    """
    if type(value) is not dict:
        raise ValueError(f"{owner} RPC payload must be a native mapping")
    payload = packb(value, use_bin_type=True, datetime=True, default=encode_extension)
    if not isinstance(payload, bytes):
        raise ValueError(f"{owner} RPC payload did not encode to bytes")
    if len(payload) > limit:
        raise ValueError(f"{owner} RPC payload exceeds its byte allowance")
    return payload


def decode_envelope(payload: bytes, limit: int, owner: str) -> dict:
    """Decode one envelope of at most `limit` bytes into a native string-key mapping.

    Raises ValueError for a non-bytes payload, an oversized payload, an unregistered or malformed extension or a decoded
    value that is not a string-key dict.
    """
    if type(payload) is not bytes:
        raise ValueError(f"{owner} RPC payload must be native bytes")
    if len(payload) > limit:
        raise ValueError(f"{owner} RPC payload exceeds its byte allowance")
    try:
        value = unpackb(
            payload,
            raw=False,
            strict_map_key=True,
            timestamp=3,
            ext_hook=decode_extension,
            max_str_len=limit,
            max_bin_len=limit,
            max_ext_len=limit,
            # An array or map holds no more items than the message has bytes.
            max_array_len=len(payload),
            max_map_len=len(payload),
        )
    except OverflowError as err:
        raise ValueError(f"{owner} RPC timestamp is outside the supported datetime range") from err
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{owner} RPC envelope requires a native string-key mapping")
    return value
