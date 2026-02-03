# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:46:26 2019

@author: syuntoku
"""

import adsk, os
from xml.etree.ElementTree import Element, SubElement
from . import Link, Joint, launch_templates
from ..utils import utils
import math


def rpy_to_R(roll, pitch, yaw):
    """
    URDF rpy convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rz = [
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0],
    ]
    Ry = [
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ]
    Rx = [
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr,  cr],
    ]
    return matmul3(matmul3(Rz, Ry), Rx)

def matmul3(A, B):
    return [
        [
            A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j]
            for j in range(3)
        ]
        for i in range(3)
    ]

def matT3(R):
    return [
        [R[0][0], R[1][0], R[2][0]],
        [R[0][1], R[1][1], R[2][1]],
        [R[0][2], R[1][2], R[2][2]],
    ]

def matvec3(R, v):
    return [
        R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
        R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
        R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
    ]

def vadd(a, b): return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
def vsub(a, b): return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def compute_world_link_poses(joints_dict, root_link='base_link'):
    """
    Tính pose world của mỗi link frame từ chain joint (forward kinematics),
    dựa trên joints_dict[*]['parent'], ['child'], ['xyz'], ['rpy'].
    Assumption: base_link world pose = identity.
    Returns:
        world_pose[link_name] = (R_world_link, t_world_link)
    """
    I = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
    world_pose = {root_link: (I, [0.0, 0.0, 0.0])}

    # Lặp cho tới khi không thêm được link mới
    progress = True
    while progress:
        progress = False
        for jname, jd in joints_dict.items():
            parent = jd['parent']
            child  = jd['child']
            if parent in world_pose and child not in world_pose:
                R_wp, t_wp = world_pose[parent]
                xyz = jd['xyz']
                rpy = jd.get('rpy', [0.0, 0.0, 0.0])

                R_pc = rpy_to_R(rpy[0], rpy[1], rpy[2])
                # t_wc = t_wp + R_wp * t_pc
                t_wc = vadd(t_wp, matvec3(R_wp, xyz))
                # R_wc = R_wp * R_pc
                R_wc = matmul3(R_wp, R_pc)

                world_pose[child] = (R_wc, t_wc)
                progress = True

    return world_pose

def inertia6_to_mat(I6):
    ixx, iyy, izz, ixy, iyz, ixz = I6
    return [
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ]

def inertia_mat_to_6(M):
    return [M[0][0], M[1][1], M[2][2], M[0][1], M[1][2], M[0][2]]

def rotate_inertia_to_link_frame(I_world_6, R_world_link):
    """
    I_link = R^T * I_world * R
    (giả sử I_world đã là inertia tại COM, nhưng đang biểu diễn theo world axes)
    """
    Iw = inertia6_to_mat(I_world_6)
    Rt = matT3(R_world_link)
    tmp = matmul3(Rt, Iw)
    Il = matmul3(tmp, R_world_link)
    return inertia_mat_to_6(Il)


def write_link_urdf(joints_dict, repo, links_xyz_dict, file_name, inertial_dict):
    """
    center_of_mass sẽ được quy đổi về CHILD LINK FRAME của joint tương ứng:
        com_child = R_wc^T * (com_world - t_wc)
    """
    # 1) Tính world pose của mỗi link frame theo chuỗi joint
    world_pose = compute_world_link_poses(joints_dict, root_link='base_link')

    with open(file_name, mode='a') as f:
        # base_link
        center_of_mass = inertial_dict['base_link']['center_of_mass']
        link = Link.Link(
            name='base_link', xyz=[0,0,0],
            center_of_mass=center_of_mass, repo=repo,
            mass=inertial_dict['base_link']['mass'],
            inertia_tensor=inertial_dict['base_link']['inertia']
        )
        links_xyz_dict[link.name] = link.xyz
        link.make_link_xml()
        f.write(link.link_xml)
        f.write('\n')

        # others
        for joint_name, jd in joints_dict.items():
            name = jd['child']

            # COM world (từ Fusion)
            com_world = inertial_dict[name]['center_of_mass']

            # pose child link in world (từ FK)
            if name not in world_pose:
                # Nếu tree thiếu / không resolve được, fallback: dùng COM world như cũ
                com_child = com_world
                inertia_child = inertial_dict[name]['inertia']
            else:
                R_wc, t_wc = world_pose[name]
                Rt = matT3(R_wc)
                com_child = matvec3(Rt, vsub(com_world, t_wc))

                # (khuyến nghị) xoay inertia về cùng axes của link frame
                inertia_child = rotate_inertia_to_link_frame(inertial_dict[name]['inertia'], R_wc)

            link = Link.Link(
                name=name, xyz=jd['xyz'],
                center_of_mass=com_child,
                repo=repo,
                mass=inertial_dict[name]['mass'],
                inertia_tensor=inertia_child
            )

            links_xyz_dict[link.name] = link.xyz
            link.make_link_xml()
            f.write(link.link_xml)
            f.write('\n')


def write_joint_urdf(joints_dict, repo, links_xyz_dict, file_name):
    with open(file_name, mode='a') as f:
        for j in joints_dict:
            parent = joints_dict[j]['parent']
            child = joints_dict[j]['child']
            joint_type = joints_dict[j]['type']
            upper_limit = joints_dict[j]['upper_limit']
            lower_limit = joints_dict[j]['lower_limit']
            # Use parent-relative xyz directly from joints_dict (no more parent-child subtraction)
            xyz = joints_dict[j]['xyz']
            rpy = joints_dict[j].get('rpy', [0.0, 0.0, 0.0])

            joint = Joint.Joint(name=j, joint_type = joint_type, xyz=xyz, rpy=rpy, \
            axis=joints_dict[j]['axis'], parent=parent, child=child, \
            upper_limit=upper_limit, lower_limit=lower_limit)
            joint.make_joint_xml()
            joint.make_transmission_xml()
            f.write(joint.joint_xml)
            f.write('\n')

def write_gazebo_endtag(file_name):
    """
    Write about gazebo_plugin and the </robot> tag at the end of the urdf


    Parameters
    ----------
    file_name: str
        urdf full path
    """
    with open(file_name, mode='a') as f:
        f.write('</robot>\n')


def write_urdf(joints_dict, links_xyz_dict, inertial_dict, package_name, robot_name, save_dir):
    try: os.mkdir(save_dir + '/urdf')
    except: pass

    file_name = save_dir + '/urdf/' + robot_name + '.xacro'  # the name of urdf file
    repo = package_name + '/meshes/'  # the repository of binary stl files
    with open(file_name, mode='w') as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<robot name="{}" xmlns:xacro="http://www.ros.org/wiki/xacro">\n'.format(robot_name))
        f.write('\n')
        f.write('<xacro:include filename="$(find {})/urdf/materials.xacro" />'.format(package_name))
        f.write('\n')
        f.write('<xacro:include filename="$(find {})/urdf/{}.trans" />'.format(package_name, robot_name))
        f.write('\n')
        f.write('<xacro:include filename="$(find {})/urdf/{}.gazebo" />'.format(package_name, robot_name))
        f.write('\n')

    write_link_urdf(joints_dict, repo, links_xyz_dict, file_name, inertial_dict)
    write_joint_urdf(joints_dict, repo, links_xyz_dict, file_name)
    write_gazebo_endtag(file_name)

def write_urdf_sim(joints_dict, links_xyz_dict, inertial_dict, package_name, robot_name, save_dir):
    try: os.mkdir(save_dir + '/urdf')
    except: pass

    file_name = save_dir + '/urdf/' + robot_name + '.xacro'  # the name of urdf file
    repo = package_name + '/meshes/'  # the repository of binary stl files
    with open(file_name, mode='w') as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<robot name="{}" xmlns:xacro="http://www.ros.org/wiki/xacro">\n'.format(robot_name))
        f.write('\n')
        f.write('<xacro:include filename="$(find {})/urdf/materials.xacro" />'.format(package_name))
        f.write('\n')
        f.write('<xacro:include filename="$(find {})/urdf/{}.ros2control" />'.format(package_name, robot_name))
        f.write('\n')
        f.write('<xacro:include filename="$(find {})/urdf/{}.gazebo" />'.format(package_name, robot_name))
        f.write('\n')

    write_link_urdf(joints_dict, repo, links_xyz_dict, file_name, inertial_dict)
    write_joint_urdf(joints_dict, repo, links_xyz_dict, file_name)
    write_gazebo_endtag(file_name)


def write_materials_xacro(joints_dict, links_xyz_dict, inertial_dict, package_name, robot_name, save_dir):
    try: os.mkdir(save_dir + '/urdf')
    except: pass

    file_name = save_dir + '/urdf/materials.xacro'  # the name of urdf file
    with open(file_name, mode='w') as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<robot name="{}" xmlns:xacro="http://www.ros.org/wiki/xacro" >\n'.format(robot_name))
        f.write('\n')
        f.write('<material name="silver">\n')
        f.write('  <color rgba="0.700 0.700 0.700 1.000"/>\n')
        f.write('</material>\n')
        f.write('\n')
        f.write('</robot>\n')

def write_transmissions_xacro(joints_dict, links_xyz_dict, inertial_dict, package_name, robot_name, save_dir):
    """
    Write joints and transmission information into urdf "repo/file_name"


    Parameters
    ----------
    joints_dict: dict
        information of the each joint
    repo: str
        the name of the repository to save the xml file
    links_xyz_dict: dict
        xyz information of the each link
    file_name: str
        urdf full path
    """

    file_name = save_dir + '/urdf/{}.trans'.format(robot_name)  # the name of urdf file
    with open(file_name, mode='w') as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<robot name="{}" xmlns:xacro="http://www.ros.org/wiki/xacro" >\n'.format(robot_name))
        f.write('\n')

        for j in joints_dict:
            parent = joints_dict[j]['parent']
            child = joints_dict[j]['child']
            joint_type = joints_dict[j]['type']
            upper_limit = joints_dict[j]['upper_limit']
            lower_limit = joints_dict[j]['lower_limit']
            # Use parent-relative xyz directly from joints_dict
            xyz = joints_dict[j]['xyz']

            joint = Joint.Joint(name=j, joint_type = joint_type, xyz=xyz, \
            axis=joints_dict[j]['axis'], parent=parent, child=child, \
            upper_limit=upper_limit, lower_limit=lower_limit)
            if joint_type != 'fixed':
                joint.make_transmission_xml()
                f.write(joint.tran_xml)
                f.write('\n')

        f.write('</robot>\n')


def write_ros2control_xacro(joints_dict, links_xyz_dict, inertial_dict, package_name, robot_name, save_dir):
    """
    Write joints and ros2 control information into urdf "repo/file_name"


    Parameters
    ----------
    joints_dict: dict
        information of the each joint
    repo: str
        the name of the repository to save the xml file
    links_xyz_dict: dict
        xyz information of the each link
    file_name: str
        urdf full path
    """

    file_name = save_dir + '/urdf/{}.ros2control'.format(robot_name)  # the name of urdf file
    with open(file_name, mode='w') as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<robot name="{}" xmlns:xacro="http://www.ros.org/wiki/xacro" >\n'.format(robot_name))
        f.write('\n')

        '''
        for j in joints_dict:
            parent = joints_dict[j]['parent']
            child = joints_dict[j]['child']
            joint_type = joints_dict[j]['type']
            upper_limit = joints_dict[j]['upper_limit']
            lower_limit = joints_dict[j]['lower_limit']
            try:
                xyz = [round(p-c, 6) for p, c in \
                    zip(links_xyz_dict[parent], links_xyz_dict[child])]  # xyz = parent - child
            except KeyError as ke:
                app = adsk.core.Application.get()
                ui = app.userInterface
                ui.messageBox("There seems to be an error with the connection between\n\n%s\nand\n%s\n\nCheck \
whether the connections\nparent=component2=%s\nchild=component1=%s\nare correct or if you need \
to swap component1<=>component2"
                % (parent, child, parent, child), "Error!")
                quit()

            joint = Joint.Joint(name=j, joint_type = joint_type, xyz=xyz, \
            axis=joints_dict[j]['axis'], parent=parent, child=child, \
            upper_limit=upper_limit, lower_limit=lower_limit)
            if joint_type != 'fixed':
                joint.make_transmission_xml()
                f.write(joint.tran_xml)
                f.write('\n')
        '''
        f.write('</robot>\n')



def write_gazebo_xacro(joints_dict, links_xyz_dict, inertial_dict, package_name, robot_name, save_dir):
    try: os.mkdir(save_dir + '/urdf')
    except: pass

    file_name = save_dir + '/urdf/' + robot_name + '.gazebo'  # the name of urdf file
    repo = robot_name + '/meshes/'  # the repository of binary stl files
    #repo = package_name + '/' + robot_name + '/bin_stl/'  # the repository of binary stl files
    with open(file_name, mode='w') as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<robot name="{}" xmlns:xacro="http://www.ros.org/wiki/xacro" >\n'.format(robot_name))
        f.write('\n')
        f.write('<xacro:property name="body_color" value="Gazebo/Silver" />\n')
        f.write('\n')

        gazebo = Element('gazebo')
        plugin = SubElement(gazebo, 'plugin')
        plugin.attrib = {'name':'control', 'filename':'libgazebo_ros_control.so'}
        gazebo_xml = "\n".join(utils.prettify(gazebo).split("\n")[1:])
        f.write(gazebo_xml)

        # for base_link
        f.write('<gazebo reference="base_link">\n')
        f.write('  <material>${body_color}</material>\n')
        f.write('  <mu1>0.2</mu1>\n')
        f.write('  <mu2>0.2</mu2>\n')
        f.write('  <self_collide>true</self_collide>\n')
        f.write('  <gravity>true</gravity>\n')
        f.write('</gazebo>\n')
        f.write('\n')

        # others
        for joint in joints_dict:
            name = joints_dict[joint]['child']
            f.write('<gazebo reference="{}">\n'.format(name))
            f.write('  <material>${body_color}</material>\n')
            f.write('  <mu1>0.2</mu1>\n')
            f.write('  <mu2>0.2</mu2>\n')
            f.write('  <self_collide>true</self_collide>\n')
            f.write('</gazebo>\n')
            f.write('\n')

        f.write('</robot>\n')

def write_gazebo_sim_xacro(joints_dict, links_xyz_dict, inertial_dict, package_name, robot_name, save_dir):
    try: os.mkdir(save_dir + '/urdf')
    except: pass

    file_name = save_dir + '/urdf/' + robot_name + '.gazebo'  # the name of urdf file
    repo = robot_name + '/meshes/'  # the repository of binary stl files
    #repo = package_name + '/' + robot_name + '/bin_stl/'  # the repository of binary stl files
    with open(file_name, mode='w') as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<robot name="{}" xmlns:xacro="http://www.ros.org/wiki/xacro" >\n'.format(robot_name))
        f.write('\n')
        f.write('<xacro:property name="body_color" value="Gazebo/Silver" />\n')
        f.write('\n')

        #gazebo = Element('gazebo')
        #plugin = SubElement(gazebo, 'plugin')
        #plugin.attrib = {'name':'control', 'filename':'libgazebo_ros_control.so'}
        #gazebo_xml = "\n".join(utils.prettify(gazebo).split("\n")[1:])
        #f.write(gazebo_xml)

        # for base_link
        f.write('<gazebo reference="base_link">\n')
        f.write('  <material>${body_color}</material>\n')
        f.write('  <mu1>0.2</mu1>\n')
        f.write('  <mu2>0.2</mu2>\n')
        f.write('  <self_collide>true</self_collide>\n')
        f.write('  <gravity>true</gravity>\n')
        f.write('</gazebo>\n')
        f.write('\n')

        # others
        for joint in joints_dict:
            name = joints_dict[joint]['child']
            f.write('<gazebo reference="{}">\n'.format(name))
            f.write('  <material>${body_color}</material>\n')
            f.write('  <mu1>0.2</mu1>\n')
            f.write('  <mu2>0.2</mu2>\n')
            f.write('  <self_collide>true</self_collide>\n')
            f.write('</gazebo>\n')
            f.write('\n')

        f.write('</robot>\n')



def write_display_launch(package_name, robot_name, save_dir):
    """
    write display launch file "save_dir/launch/display.launch"


    Parameter
    ---------
    robot_name: str
    name of the robot
    save_dir: str
    path of the repository to save
    """
    try: os.mkdir(save_dir + '/launch')
    except: pass

    file_text = launch_templates.get_display_launch_text(package_name, robot_name)

    file_name = os.path.join(save_dir, 'launch', 'display.launch.py')
    with open(file_name, mode='w') as f:
        f.write(file_text)

def write_gazebo_launch(package_name, robot_name, save_dir):
    """
    write gazebo launch file "save_dir/launch/gazebo.launch"


    Parameter
    ---------
    robot_name: str
        name of the robot
    save_dir: str
        path of the repository to save
    """

    try: os.mkdir(save_dir + '/launch')
    except: pass

    file_text = launch_templates.get_gazebo_launch_text(package_name, robot_name)

    file_name = os.path.join(save_dir, 'launch', 'gazebo.launch.py')
    with open(file_name, mode='w') as f:
        f.write(file_text)


def write_gazebo_sim_launch(package_name, robot_name, save_dir):
    """
    write gazebo launch file "save_dir/launch/gazebo.launch"


    Parameter
    ---------
    robot_name: str
        name of the robot
    save_dir: str
        path of the repository to save
    """

    try: os.mkdir(save_dir + '/launch')
    except: pass

    file_text = launch_templates.get_gazebo_sim_launch_text(package_name, robot_name)

    file_name = os.path.join(save_dir, 'launch', 'gazebo.launch.py')
    with open(file_name, mode='w') as f:
        f.write(file_text)
