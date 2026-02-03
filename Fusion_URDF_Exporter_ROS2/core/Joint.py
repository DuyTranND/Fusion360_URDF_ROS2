# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:17:17 2019

@author: syuntoku
"""

import adsk, re, os, math
from datetime import datetime
from xml.etree.ElementTree import Element, SubElement
from ..utils import utils
from typing import Dict, Any, Tuple, List

class Joint:
    def __init__(self, name, xyz, axis, parent, child, joint_type, upper_limit, lower_limit, rpy=None):
        """
        Attributes
        ----------
        name: str
            name of the joint
        type: str
            type of the joint(ex: rev)
        xyz: [x, y, z]
            coordinate of the joint
        axis: [x, y, z]
            coordinate of axis of the joint
        parent: str
            parent link
        child: str
            child link
        joint_xml: str
            generated xml describing about the joint
        tran_xml: str
            generated xml describing about the transmission
        """
        self.name = name
        self.type = joint_type
        self.xyz = xyz
        self.parent = parent
        self.child = child
        self.joint_xml = None
        self.tran_xml = None
        self.axis = axis  # for 'revolute' and 'continuous'
        self.upper_limit = upper_limit  # for 'revolute' and 'prismatic'
        self.lower_limit = lower_limit  # for 'revolute' and 'prismatic'
        
        self.rpy = rpy if rpy is not None else [0.0, 0.0, 0.0]
        
    def make_joint_xml(self):
        """
        Generate the joint_xml and hold it by self.joint_xml
        """
        joint = Element('joint')
        joint.attrib = {'name':self.name, 'type':self.type}
        
        origin = SubElement(joint, 'origin')
        origin.attrib = {'xyz':' '.join([str(_) for _ in self.xyz]), 'rpy': ' '.join([str(_) for _ in self.rpy])}
        parent = SubElement(joint, 'parent')
        parent.attrib = {'link':self.parent}
        child = SubElement(joint, 'child')
        child.attrib = {'link':self.child}
        if self.type == 'revolute' or self.type == 'continuous' or self.type == 'prismatic':        
            axis = SubElement(joint, 'axis')
            axis.attrib = {'xyz':' '.join([str(_) for _ in self.axis])}
        if self.type == 'revolute' or self.type == 'prismatic':
            limit = SubElement(joint, 'limit')
            limit.attrib = {'upper': str(self.upper_limit), 'lower': str(self.lower_limit),
                            'effort': '100', 'velocity': '100'}
            
        self.joint_xml = "\n".join(utils.prettify(joint).split("\n")[1:])

    def make_transmission_xml(self):
        """
        Generate the tran_xml and hold it by self.tran_xml
        
        
        Notes
        -----------
        mechanicalTransmission: 1
        type: transmission interface/SimpleTransmission
        hardwareInterface: PositionJointInterface        
        """        
        
        tran = Element('transmission')
        tran.attrib = {'name':self.name + '_tran'}
        
        joint_type = SubElement(tran, 'type')
        joint_type.text = 'transmission_interface/SimpleTransmission'
        
        joint = SubElement(tran, 'joint')
        joint.attrib = {'name':self.name}
        hardwareInterface_joint = SubElement(joint, 'hardwareInterface')
        hardwareInterface_joint.text = 'hardware_interface/EffortJointInterface'
        
        actuator = SubElement(tran, 'actuator')
        actuator.attrib = {'name':self.name + '_actr'}
        hardwareInterface_actr = SubElement(actuator, 'hardwareInterface')
        hardwareInterface_actr.text = 'hardware_interface/EffortJointInterface'
        mechanicalReduction = SubElement(actuator, 'mechanicalReduction')
        mechanicalReduction.text = '1'
        
        self.tran_xml = "\n".join(utils.prettify(tran).split("\n")[1:])
import math
from typing import List, Tuple

Vec3 = List[float]
Mat3 = List[List[float]]


def split_rt_fusion_row_major(M: List[float]) -> Tuple[Mat3, Vec3]:
    """
    Fusion Transform.asArray() returns 16 floats in row-major.
    Rotation is:
        [[M0, M1, M2],
         [M4, M5, M6],
         [M8, M9, M10]]
    Translation is [M3, M7, M11]
    """
    R = [
        [M[0],  M[1],  M[2]],
        [M[4],  M[5],  M[6]],
        [M[8],  M[9],  M[10]],
    ]
    t = [M[3], M[7], M[11]]
    return R, t


def mat3_T(R: Mat3) -> Mat3:
    return [
        [R[0][0], R[1][0], R[2][0]],
        [R[0][1], R[1][1], R[2][1]],
        [R[0][2], R[1][2], R[2][2]],
    ]


def mat3_mul(A: Mat3, B: Mat3) -> Mat3:
    return [
        [
            A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]
            for j in range(3)
        ]
        for i in range(3)
    ]


def mat3_vec(R: Mat3, v: Vec3) -> Vec3:
    return [
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    ]


def vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def rotmat_to_rpy_zyx(R: Mat3, eps: float = 1e-9) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to URDF rpy (roll, pitch, yaw),
    with convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)

    Handles gimbal lock near pitch = +/- pi/2 by choosing roll = 0
    and solving yaw stably from the matrix.
    """
    # pitch = asin(-R[2][0])
    sp = clamp(-R[2][0])
    pitch = math.asin(sp)
    cp = math.cos(pitch)

    if abs(cp) > eps:
        roll = math.atan2(R[2][1], R[2][2])
        yaw = math.atan2(R[1][0], R[0][0])
        return roll, pitch, yaw

    # Gimbal lock: pitch ~ +/- pi/2
    roll = 0.0
    if sp < 0:  # pitch ~ -pi/2
        pitch = -math.pi / 2
        yaw = math.atan2(-R[0][1], -R[0][2])
    else:       # pitch ~ +pi/2
        pitch = math.pi / 2
        yaw = math.atan2(-R[0][1],  R[0][2])
    return roll, pitch, yaw


def compute_T_parent_child(M_one_child_world: List[float],
                           M_two_parent_world: List[float]) -> Tuple[Mat3, Vec3]:
    """
    Compute parent->child rigid transform from Fusion matrices:
      T_pc = inv(T_wp) * T_wc

    Returns:
      R_pc (3x3), t_pc (3,)
    """
    R_child, t_child = split_rt_fusion_row_major(M_one_child_world)
    R_parent, t_parent = split_rt_fusion_row_major(M_two_parent_world)

    R_parent_T = mat3_T(R_parent)

    # R_pc = R_parent^T * R_child
    R_pc = mat3_mul(R_parent_T, R_child)

    # t_pc = R_parent^T * (t_child - t_parent)
    dt = vec_sub(t_child, t_parent)
    t_pc = mat3_vec(R_parent_T, dt)

    return R_pc, t_pc


def compute_urdf_origin_xyz_rpy(M_one_child_world: List[float],
                               M_two_parent_world: List[float],
                               unit_scale: float = 0.01) -> Tuple[Vec3, Vec3]:
    """
    Compute URDF <origin xyz="" rpy=""> from Fusion transforms.

    - xyz is parent->child translation in meters (default cm->m with unit_scale=0.01)
    - rpy is (roll, pitch, yaw) in radians (URDF ZYX convention)
    """
    R_pc, t_pc = compute_T_parent_child(M_one_child_world, M_two_parent_world)

    xyz_m = [t_pc[0] * unit_scale, t_pc[1] * unit_scale, t_pc[2] * unit_scale]
    roll, pitch, yaw = rotmat_to_rpy_zyx(R_pc)
    rpy = [roll, pitch, yaw]

    return xyz_m, rpy

def make_joints_dict(root, msg, save_dir=None, unit_scale=0.01, axis_is_world=True):
    joint_type_list = [
        'fixed', 'revolute', 'prismatic', 'Cylinderical',
        'PinSlot', 'Planner', 'Ball'
    ]

    joints_dict: Dict[str, Dict[str, Any]] = {}

    def vec_norm(v: List[float]) -> float:
        return (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) ** 0.5

    def vec_unit(v: List[float]) -> List[float]:
        n = vec_norm(v)
        if n < 1e-12:
            return [0.0, 0.0, 0.0]
        return [v[0]/n, v[1]/n, v[2]/n]

    for joint in root.joints:
        joint_dict: Dict[str, Any] = {}

        joint_type = joint_type_list[joint.jointMotion.jointType]
        joint_dict['type'] = joint_type

        # parent/child naming
        if joint.occurrenceTwo.component.name == 'base_link':
            joint_dict['parent'] = 'base_link'
        else:
            joint_dict['parent'] = re.sub(r'[ :()]', '_', joint.occurrenceTwo.name)
        joint_dict['child'] = re.sub(r'[ :()]', '_', joint.occurrenceOne.name)

        # ---- compute origin xyz + rpy from transforms (parent->child) ----
        try:
            M_one = joint.occurrenceOne.transform.asArray()  # child in world
            M_two = joint.occurrenceTwo.transform.asArray()  # parent in world

            xyz_m, rpy = compute_urdf_origin_xyz_rpy(M_one, M_two, unit_scale=unit_scale)
            joint_dict['xyz'] = [round(xyz_m[0], 6), round(xyz_m[1], 6), round(xyz_m[2], 6)]
            joint_dict['rpy'] = [round(rpy[0], 12), round(rpy[1], 12), round(rpy[2], 12)]

            # For axis conversion if axis is in world:
            R_child, _ = split_rt_fusion_row_major(M_one)

        except Exception as e:
            msg = f"{joint.name} transform read failed: {type(e).__name__}: {e}"
            break

        # defaults
        joint_dict['axis'] = [0.0, 0.0, 0.0]
        joint_dict['upper_limit'] = 0.0
        joint_dict['lower_limit'] = 0.0

        # ---- axis + limits ----
        if joint_type == 'revolute':
            axis_raw = list(joint.jointMotion.rotationAxisVector.asArray())

            if axis_is_world:
                # world -> child(local) : axis_local = R_child^T * axis_world
                axis_local = mat3_vec(mat3_T(R_child), axis_raw)
            else:
                axis_local = axis_raw

            axis_local = vec_unit(axis_local)
            joint_dict['axis'] = [round(axis_local[0], 6), round(axis_local[1], 6), round(axis_local[2], 6)]

            max_enabled = joint.jointMotion.rotationLimits.isMaximumValueEnabled
            min_enabled = joint.jointMotion.rotationLimits.isMinimumValueEnabled
            if max_enabled and min_enabled:
                joint_dict['upper_limit'] = round(joint.jointMotion.rotationLimits.maximumValue, 6)
                joint_dict['lower_limit'] = round(joint.jointMotion.rotationLimits.minimumValue, 6)
            elif max_enabled and not min_enabled:
                msg = joint.name + ' is not set its lower limit. Please set it and try again.'
                break
            elif not max_enabled and min_enabled:
                msg = joint.name + ' is not set its upper limit. Please set it and try again.'
                break
            else:
                joint_dict['type'] = 'continuous'

        elif joint_type == 'prismatic':
            axis_raw = list(joint.jointMotion.slideDirectionVector.asArray())

            if axis_is_world:
                axis_local = mat3_vec(mat3_T(R_child), axis_raw)
            else:
                axis_local = axis_raw

            axis_local = vec_unit(axis_local)
            joint_dict['axis'] = [round(axis_local[0], 6), round(axis_local[1], 6), round(axis_local[2], 6)]

            max_enabled = joint.jointMotion.slideLimits.isMaximumValueEnabled
            min_enabled = joint.jointMotion.slideLimits.isMinimumValueEnabled
            if max_enabled and min_enabled:
                joint_dict['upper_limit'] = round(joint.jointMotion.slideLimits.maximumValue * unit_scale, 6)
                joint_dict['lower_limit'] = round(joint.jointMotion.slideLimits.minimumValue * unit_scale, 6)
            elif max_enabled and not min_enabled:
                msg = joint.name + ' is not set its lower limit. Please set it and try again.'
                break
            elif not max_enabled and min_enabled:
                msg = joint.name + ' is not set its upper limit. Please set it and try again.'
                break

        elif joint_type == 'fixed':
            pass

        joints_dict[joint.name] = joint_dict

    return joints_dict, msg

