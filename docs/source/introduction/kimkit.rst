.. _kimkit:

KIMkit Setup
============

KIMkit is a standalone python package implementing an
Interatomic Model management and storage system based upon
and intended to be compatible with the standards set out by the
OpenKIM Project https://openkim.org/. KIMkit provides methods to store, archive, edit,
and track changes to Interatomic Models, which are simulation codes used
to compute specific interactions between atoms, e.g. an interatomic
potential or force field.

Installing from PyPi
--------------------

To install the latest release from the KIMkit PyPi repositiory simply issue a

.. code-block:: console

     $ pip install kimkit


This will be done automatically if installing the orchestrator's dependecies listed in pyproject.toml.

When KIMkit installs itself as a pip package, it will create a /kimkit/ directory under the user's home directory.
Inside this will be configuration files, log files, and the default location for the KIMkit repository where the items are stored.
There will be a file called KIMkit-env, that contains paths and other settings needed for KIMkit to function correctly.
By editing these settings users may configure the behavior of KIMkit, where items are stored, add the credentials to connect to mongodb, etc.

Installing From Source
----------------------

To clone the git repo, use the https option:

.. code-block:: console

     $ git clone git@github.com:openkim/KIMkit.git kimkit

.. note::
    To name the repo something other than kimkit, replace the trailing ``kimkit`` token with the desired (path) name.

You can switch to other branches using the ``git checkout`` command. To check for updates to the code, use ``git fetch``. If the branch has been updated, you can update your local version by using the ``git pull`` command. To see all available branches, you can check the repo online, or enter ``git branch -a`` from the command line.

Then, to install navigate to the root /kimkit/ directory that was cloned and issue

.. code-block:: console

    $ pip install .


You may add the -e option to make the resulting kimkit repository editable, if you want to make frequent changes to the KIMkit source code.

Dependencies
------------

MongoDB: KIMkit maintains a mongodb database of metadata about the interatomic potentials
stored in its repository to allow for querying and make relationships between
items and their drivers more transparent. This must be installed seperately from
kimkit itself as it isn't a python package. Installation instrutions for your
operating sysem are available at https://www.mongodb.com/docs/manual/installation/

The main branch of KIMkit is designed to
work out of the box with a new MongoDB instance on the default settings.
If more specific configuration is required, additional parameters may be
saved in a file called KIMkit-env placed in the kimkit source directory
(/home/User/kimkit/KIMkit-env or ./kimkit/kimkit/KIMkit-env ), which will be parsed by the config
module. Those parameters can then be used to connect to a MONGODB instance
using the code below.

.. code-block::

    user = config.MONGODB_USERNAME
    password = configf.MONGODB_PASSWORD
    host = config.MONGODB_HOSTNAME
    port = config.MONGODB_PORT
    db_name = config.MONGODB_DATABASE

    # example arg
    args = "ssl=true&tlsAllowInvalidCertificates=true"

    db_uri = "mongodb://%s:%s@%s:%s/%s?%s" % (user, password, host, port, db_name, args)

    client = pymongo.MongoClient(host=db_uri)

    db = client[cf.MONGODB_DATABASE]


Post-Install Setup
------------------

Inside your /kimkit/ settings directory there is a file called editors.txt, with permissions 644
(only the administrator has write permissions).

The Administrator or Editors may promote users to editors by running

::
    kimkit.users.add_editor("username_of_new_editor",run_as_administrator=True)

General users of kimkit can only modify or edit content that they contributed to,
to prevent unintentional deletion of another user's work. However, this is sometimes
required. Editors with their usernames listed in editors.txt may perform destructive
operations on other's content in kimkit by passing the argument run_as_editor=True.

First Time Using KIMkit
-----------------------

In order to track contributions to kimkit, users must be assigned a UUID before
they can contribute. If you haven't already, kimkit will prompt you to add yourself
to the database of users and be assigned a UUID by running

::
    kimkit.users.add_self_as_user("John Doe (your real name)")

Once you have been assigned a UUID within a given installation of KIMkit it will
be automatically retrieved from the users collection in the database based on your username
when editing content in kimkit, and your assigned UUID will be recorded in metadata
of any kimkit items you've contributed.
