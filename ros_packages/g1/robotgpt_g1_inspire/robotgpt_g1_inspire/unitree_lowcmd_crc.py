"""
Function for computing the CRC of a unitree_ros2 LowCmd message.

The code in this script is converted from C++ code found in:
https://github.com/unitreerobotics/unitree_ros2/blob/master/example/src/src/common/motor_crc_hg.cpp
The unitree_ros2 repository is licensed under:

BSD 3-Clause License

Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import struct

POLY = 0x04C11DB7


def get_unitree_lowcmd_crc(msg):
    data = bytearray()

    # LowCmd header
    data += struct.pack(
        "<BBxx",
        msg.mode_pr,
        msg.mode_machine,
    )

    # 35 motors
    for m in msg.motor_cmd:
        data += struct.pack(
            "<Bxxx5fI",
            m.mode,
            m.q,
            m.dq,
            m.tau,
            m.kp,
            m.kd,
            m.reserve,
        )

    # LowCmd reserve[4]
    data += struct.pack("<4I", *msg.reserve)

    assert len(data) == 1000

    crc = 0xFFFFFFFF

    for i in range(0, len(data), 4):
        word = struct.unpack_from("<I", data, i)[0]

        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ POLY
            else:
                crc = (crc << 1) & 0xFFFFFFFF

            if word & 0x80000000:
                crc ^= POLY

            word = (word << 1) & 0xFFFFFFFF

    return crc
