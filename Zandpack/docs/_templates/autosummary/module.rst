{#-
  Module page for the API reference.

  The module docstring is rendered on its own (``:no-members:``) and the
  classes/functions defined by the module are listed below it. Every listed
  member has a page of its own -- generated with ``:toctree:`` -- so the detail
  lives there instead of being repeated inline here. Module-level attributes are
  listed for overview but are not given separate pages.
-#}
{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:

{%- if attributes %}

.. rubric:: Module Attributes

.. autosummary::
   :nosignatures:

{% for item in attributes %}
   {{ item }}
{%- endfor %}
{%- endif %}

{%- if exceptions %}

.. rubric:: Exceptions

.. autosummary::
   :toctree: {{ objname }}

{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{%- endif %}

{%- if classes %}

.. rubric:: Classes

.. autosummary::
   :toctree: {{ objname }}

{% for item in classes %}
   {{ item }}
{%- endfor %}
{%- endif %}

{%- if functions %}

.. rubric:: Functions

.. autosummary::
   :toctree: {{ objname }}

{% for item in functions %}
   {{ item }}
{%- endfor %}
{%- endif %}

{%- if modules %}

.. rubric:: Modules

.. autosummary::
   :toctree: {{ objname }}
   :recursive:

{% for item in modules %}
   {{ item }}
{%- endfor %}
{%- endif %}
