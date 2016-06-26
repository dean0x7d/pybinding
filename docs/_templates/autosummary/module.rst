{{ objname }}
{{ underline }}

.. automodule:: {{ fullname }}
    :members: {{ (functions + classes)|join(', ') }}

    {% if (functions + classes + exceptions)|length > 1 %}

    {% block classes %}
    {% if classes %}
    .. rubric:: Classes

    .. autosummary::
    {% for item in classes %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block functions %}
    {% if functions %}
    .. rubric:: Functions

    .. autosummary::
    {% for item in functions %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block exceptions %}
    {% if exceptions %}
    .. rubric:: Exceptions

    .. autosummary::
    {% for item in exceptions %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% endif %}
