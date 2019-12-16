from setuptools import setup, find_packages

setup(
    name='is_gesture_recognizier',
    version='0.0.1',
    description='',
    url='http://github.com/luizcarloscf/is-gesture-recognizier',
    author='luizcarloscf',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'is-gesture-recognizierr=is_gesture_recognizier.service:main',
        ],
    },
    zip_safe=False,
    install_requires=[
        'is-wire==1.2.0',
        'is-msgs==1.1.10',
        'numpy==1.16.1',
        'opencensus-ext-zipkin==0.2.1',
        'python-dateutil==2.8.0',
    ],
)