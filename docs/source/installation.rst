Installation
============

.. _installation:

Here we provide a basic guide to install Meep on your computer running Windows or MacOS. Note that this is only one way to install, and if you need any special installation or face some sort of problems not addressed here, you should see the installation instructions in the official Meep documentation `here <https://meep.readthedocs.io/en/master/Installation/#>`_. 

First, we provide a table where you can look which steps are necessary for your operating system. 

.. list-table:: Installation steps on Windows and MacOS
   :widths: 20 10 10
   :header-rows: 1

   * - Installation step
     - Windows
     - MacOS
   * - Windows Subsystem for Linux (WSL)
     - Yes
     - No
   * - Conda
     - Yes
     - Yes
   * - Pymeep 
     - Yes
     - Yes


Installation of Windows Subsystem for Linux
----------------------
The first step of the process on Windows is to install `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install/>`_ or WSL,
as a native Windows installation is not supported for MEEP. To install WSL, simply right-click on your Windows Command Prompt, 
select "Run as administrator" and run the command: 

    ``wsl --install`` 

Once the command prompt tells you the installation is complete, restart your computer. Once your computer has restarted the Ubuntu command prompt should automatically be open
and finishing the installation. It will ask you to create a UNIX user account with a name and password. Make sure to remember these for later use, incase the system requires you to log in again at a later date.
Once the WSL installation is finished, if you check the left sidebar of your file explorer, you should see the Linux library with its penguin logo. This is where all relevant files and folders
will be located once you begin using MEEP.


Installation of Conda
----------------------

The next step is to install the package and enviroment manager Conda. In this guide, we install `Miniconda <https://docs.anaconda.com/miniconda/>`_ since it will include everything necessary to work with MEEP on Python. You can install Miniconda using the graphical installers provided via the link. We also provide here a detailed guide for the quick installation via the command prompt.

All the commands used to install and use Miniconda and MEEP will be run on the Ubuntu-commandprompt (on WSL) and on the terminal (on MacOS). Ubuntu command prompt can be found by searching for 'Ubuntu' on the Windows search. On MacOS, you can find the corresponding command prompt by opening the Spotlight search with cmd + space and search for 'Terminal'.

.. note::
    To copy-paste commands into the Ubuntu-command prompt we first CTRL + C normally on the Windows side and then simply right-click in the Ubuntu-command prompt to paste. 
    If this does not work, try using CTRL + SHIFT + V or clicking the logo at the top-left, selecting properties and checking the box that says "QuickEdit Mode".


First, you should create the directory in which to install Miniconda. This happens by running the commmand

    ``mkdir -p ~/miniconda3``

Next, you should obtain the the installer for your system:

* On WSL: ``wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh``
* On MacOS (Apple Silicon): ``curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh``
* On MacOS (Intel): ``curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/miniconda3/miniconda.sh``

Once this has finished running, you can perform the installation by the command
    ``bash miniconda.sh -b -u -p ~/miniconda3/miniconda.sh``

.. note::
    This command performs a so-called silent install. This means you accept `the Anaconda's Terms of Service <https://legal.anaconda.com/policies/en/>`_ by default.

Next we add the location of Miniconda to the variable PATH so that we can use 
the installed Miniconda later. This is done by the commmand
    ``export PATH=~/miniconda3/bin:$PATH``


Installing Pymeep
-----------------

Next we install the Python interface of MEEP using Conda. The command to run is

    ``conda create -n mp -c conda-forge pymeep pymeep-extras``

.. note::
    With a Mac with Apple silicon chip (M series), it might be necessary to use the line 
        ``CONDA_SUBDIR=osx-64 conda create -n mp -c conda-forge pymeep pymeep-extras``
    This is because there are installers of Pymeep for only ``linux-64`` and ``os-x64`` provided by Conda.


This will create a Python environment called "mp". Next we need to activate the environment with the command

    ``conda activate mp``

Ubuntu may require you to initialize the environment first. If so, simply input the requested initialization command and then rerun the activation command.
Please note that the activation command will have to be run everytime you restart your commandprompt and want to start working in the MEEP environment.
Lastly we will import the meep library for python by running the command

    ``python -c 'import meep'``

Now any Python script files containing MEEP simulation code can simply be run by typing ``python SIMULATION_FILENAME.py``. If you are new to using and navigating filesystems in a Linux commandprompt, also known as a BASH Shell,
the necessary commands to achieve most relevant tasks can be found `here <https://www.educative.io/blog/bash-shell-command-cheat-sheet>`_.

