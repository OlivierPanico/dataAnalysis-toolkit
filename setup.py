from setuptools import setup, find_packages

setup(
    name='dataAnalysis',
    version='1.0.0',
    author='OP',
    author_email='olivier.panico@free.fr',
    description='',
    packages=find_packages(),  # Automatically discover and include all packages
    install_requires=[
        # List the dependencies your package needs
        'numpy',
        'scipy',
        'matplotlib'
    ],
    classifiers=[
        # 'Development Status :: 3 - Alpha',
        # 'Intended Audience :: Developers',
        # 'License :: OSI Approved :: MIT License',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: 3.7',
        # 'Programming Language :: Python :: 3.8',
        # Add other classifiers as needed
    ],
    entry_points={
        'console_scripts': [
            # Define any command-line scripts your package provides
        ],
    },
)