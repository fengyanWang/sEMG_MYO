#coding:UTF-8
from __future__ import print_function
from enum import Enum
import struct
import time
import re
import serial
from serial.tools.list_ports import comports


class VibrationType(Enum):
    NONE = 0
    SHORT = 1
    MEDIUM = 2
    LONG = 3

class PoseType(Enum):
    REST = 0
    FIST = 1
    WAVE_IN = 2
    WAVE_OUT = 3
    FINGERS_SPREAD = 4
    DOUBLE_TAP = 5
    UNKNOWN = 255

class Packet(object):
    def __init__(self, ords):
        self.data_type = ords[0]
        self.cls = ords[2]
        self.command = ords[3]
        self.payload = multichr(ords[4:])

    def __repr__(self):
        return 'Packet(%02X, %02X, %02X, [%s])' % \
            (self.data_type, self.cls, self.command,
             ' '.join('%02X' % b for b in multiord(self.payload)))
def pack(fmt, *args):
    return struct.pack('<' + fmt, *args)

def unpack(fmt, *args):
    return struct.unpack('<' + fmt, *args)

def multichr(values):
    return ''.join(map(chr, values))

def multiord(values):
    return map(ord, values)


class BLE(object):
    def __init__(self, tty_port):
        self.ser = serial.Serial(port=tty_port, baudrate=9600, timeout=1, dsrdtr=1)
        self.buffer = []
        self.listeners = []

    def receive_packet(self, timeout=None):
        start_time = time.time()
        self.ser.timeout = None
        while timeout is None or time.time() < start_time + timeout:
            if timeout is not None: self.ser.timeout = start_time + timeout - time.time()
            x = self.ser.read()
            if not x: return None
            packet = self.process_byte(ord(x))
            if packet:
                if packet.data_type == 0x80:
                    self.notify_event(packet)
                return packet
    def process_byte(self, x):
        if not self.buffer:
            if x in [0x00, 0x80, 0x08, 0x88]:
                self.buffer.append(x)
            return None
        elif len(self.buffer) == 1:
            self.buffer.append(x)
            self.packet_length = 4 + (self.buffer[0] & 0x07) + self.buffer[1]
            return None
        else:
            self.buffer.append(x)
        if self.packet_length and len(self.buffer) == self.packet_length:
            packet = Packet(self.buffer)
            self.buffer = []
            return packet
        return None

    def notify_event(self, p):
        for listener in self.listeners:
            if listener.__class__.__name__ == 'function':
                listener(p)
            else:
                listener.handle_data(p)

    def add_listener(self, listener):
        self.listeners.append(listener)

    def remove_listener(self, listener):
        try: 
            self.listeners.remove(listener)
        except ValueError: 
            pass

    def wait_event(self, cls, command):
        response = [None]
        def valid_packet(packet):
            if packet.cls == cls and packet.command == command:
                response[0] = packet
        self.add_listener(valid_packet)
        while response[0] is None:
            self.receive_packet()
        self.remove_listener(valid_packet)
        return response[0]

    def connect(self, address):
        return self.send_command(6, 3, pack('6sBHHHH', multichr(address), 0, 6, 6, 64, 0))

    def start_scan(self):
        return self.send_command(6, 2, b'\x01')

    def end_scan(self):
        return self.send_command(6, 4)

    def read_attribute(self, connection, attribute):
        self.send_command(4, 4, pack('BH', connection, attribute))
        return self.wait_event(4, 5)

    def write_attribute(self, connection, attribute, value):
        self.send_command(4, 5, pack('BHB', connection, attribute, len(value)) + value)
        return self.wait_event(4, 1)

    def send_command(self, cls, cmd, payload=b''):
        package = pack('4B', 0, len(payload), cls, cmd) + payload
        self.ser.write(package)
        while True:
            packet = self.receive_packet()
            if packet.data_type == 0: 
                return packet
            self.notify_event(packet)
    def disconnect(self, h):
        return self.send_command(3, 0, pack('B', h))


class Myo(object):

    def __init__(self):
        self.ble = None
        self.connection = None

    def connect(self, tty_port = None):
        
        self.safely_disconnect()
        self.find_bluetooth_adapter(tty_port)

        address = self.find_myo_device()
        connection_packet = self.ble.connect(address)
        self.connection = multiord(connection_packet.payload)[-1]
        self.ble.wait_event(3, 0)
        print('Connected.')

        is_fw_valid = self.valid_firmware_version()

        if is_fw_valid:
            device_name = self.read_attribute(0x03)
            print('Device name: %s' % device_name.payload[5:])
            self.write_attribute(0x1d, b'\x01\x00')
            self.write_attribute(0x24, b'\x02\x00')
            self.initialize()
        else:
            raise ValueError('The firmware version must be v1.x or greater.')

    def find_bluetooth_adapter(self, tty_port = None):

        if tty_port is None:
            tty_port = self.find_tty()
        if tty_port is None:
            raise ValueError('Bluetooth adapter not found!')
        self.ble = BLE(tty_port)

    def find_tty(self):
        for port in comports():
            if re.search(r'PID=2458:0*1', port[2]):
                return port[0]

        return None

    def run(self, timeout=None):
        if self.connection is not None:
            self.ble.receive_packet(timeout)
        else:
            raise ValueError('Myo device not paired.')

    def valid_firmware_version(self):
        firmware = self.read_attribute(0x17)
        _, _, _, _, major, minor, patch, build = unpack('BHBBHHHH', firmware.payload)

        print('Firmware version: %d.%d.%d.%d' % (major, minor, patch, build))

        return major > 0

    def add_listener(self, listener):
        if self.ble is not None:
            self.ble.add_listener(listener)
        else:
            print('Connect function must be called before adding a listener.')

    def vibrate(self, duration):

        cmd = b'\x03\x01'
        if duration == VibrationType.LONG:
            cmd = cmd + b'\x03'
        elif duration == VibrationType.MEDIUM:
            cmd = cmd + b'\x02'
        elif duration == VibrationType.SHORT:
            cmd = cmd + b'\x01'
        else:
            cmd = cmd + b'\x00'
            
        self.write_attribute(0x19, cmd)

    def initialize(self):
        self.write_attribute(0x28, b'\x01\x00')
        self.write_attribute(0x19, b'\x01\x03\x01\x01\x00')
        self.write_attribute(0x19, b'\x01\x03\x01\x01\x01')

    def find_myo_device(self):
        print('Find Myo device...')
        address = None
        self.ble.start_scan()
        while True:
            packet = self.ble.receive_packet()

            if packet.payload.endswith(b'\x06\x42\x48\x12\x4A\x7F\x2C\x48\x47\xB9\xDE\x04\xA9\x01\x00\x06\xD5'):
                address = list(multiord(packet.payload[2:8]))
                break

        self.ble.end_scan()
        return address

    def write_attribute(self, attribute, value):
        if self.connection is not None:
            self.ble.write_attribute(self.connection, attribute, value)

    def read_attribute(self, attribute):
        if self.connection is not None:
            return self.ble.read_attribute(self.connection, attribute)
        return None

    def safely_disconnect(self):
        if self.ble is not None:
            self.ble.end_scan()
            self.ble.disconnect(0)
            self.ble.disconnect(1)
            self.ble.disconnect(2)
            self.disconnect()

    def disconnect(self):

        if self.connection is not None:
            self.ble.disconnect(self.connection)

class DeviceListener(object):

    def handle_data(self, data):
        if data.cls != 4 and data.command != 5:
            return
        connection, attribute, data_type = unpack('BHB', data.payload[:4])
        payload = data.payload[5:]
        if attribute == 0x23:
            data_type, value, address, _, _, _ = unpack('6B', payload)
            if data_type == 3:
                self.on_pose( value)
        elif attribute == 0x27:
            vals = unpack('8HB', payload)
            emg = vals[:8]
            moving = vals[8]
            self.on_emg(emg, moving)
        elif attribute == 0x1c:
            vals = unpack('10h', payload)
            quat = vals[:4]
            acc = vals[4:7]
            gyro = vals[7:10]
            self.on_imu(quat, acc, gyro)
            
    def on_pose(self, pose):
        pass
    def on_emg(self , emg, moving):
        pass
    def on_imu(self , quat, acc, gyro ):
        pass
