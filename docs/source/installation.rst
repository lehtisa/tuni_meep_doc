Installation
============

.. _installation:


Installation on Windows
----------------------
The first step of the process is to install `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install/>`_ or WSL,
as a native Windows installation is not supported for MEEP. To install WSL, simply right-click on your Windows Command Prompt, 
select "Run as administrator" and run the command: 

    ``wsl --install`` 

Once the commandprompt tells you the installation is complete, restart your computer. Once your computer has restarted the Ubuntu commandprompt should automatically be open
and finishing the installation. It will ask you to create a UNIX user account with a name and password. Make sure to remember these for later use, incase the system requires you to log in again at a later date.
Once the WSL installation is finished, if you check the left sidebar of your file explorer, you should see the Linux library with its penguin logo. This is where all relevant files and folders
will be located once you begin using MEEP.


The next step is to install `Miniconda <https://docs.anaconda.com/miniconda/>`_. This will include everything necessary to work with MEEP on Python.
All the commands used to install and use Miniconda and MEEP will be run on the Ubuntu-commandprompt, which if closed can be found by searching for 'Ubuntu' on the Windows search.
To copy-paste commands into the Ubuntu-commandprompt we first CTRL + C normally on the Windows side and then simply right-click in the Ubuntu-commandprompt to paste.
If this does not work, try using CTRL + SHIFT + V or clicking the logo at the top-left, selecting properties and checking the box that says "QuickEdit Mode".
The first command to run is 

    ``wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh``

Once this has finished running the second command to run is

    ``bash miniconda.sh -b -p <desired_prefix>``

Here ``<desired_prefix>`` should be replaced with whatever filename you choose. The next command to run is

    ``export PATH=<desired_prefix>/bin:$PATH``

Please note the usage of the previously chosen prefix again. The next command to run is

    ``conda create -n mp -c conda-forge pymeep pymeep-extras``

This will create a python environment called "mp". Next we need to activate the environment with the command

    ``conda activate mp``

Ubuntu may require you to initialize the environment first. If so, simply input the requested initialization command and then rerun the activation command.
Please note that the activation command will have to be run everytime you restart your commandprompt and want to start working in the MEEP environment.
Lastly we will import the meep library for python by running the command

    ``python -c 'import meep'``

Now any Python script files containing MEEP simulation code can simply be run by typing ``python SIMULATION_FILENAME.py``. If you are new to using and navigating filesystems in a Linux commandprompt, also known as a BASH Shell,
the necessary commands to achieve most relevant tasks can be found `here <https://www.educative.io/blog/bash-shell-command-cheat-sheet>`_.


Installation on MacOS
---------------------
This is how to install MEEP on MacOS.