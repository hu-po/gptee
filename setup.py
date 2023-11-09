from setuptools import find_packages, setup

package_name = 'gptee'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hu-po',
    entry_points={
        'console_scripts': [
            'gptee = gptee.gptee:main'
        ],
    },
)
