---
layout: default
title: Home
---

## Welcome

I’m German W Aparicio Jr. This site shares my CV, contact details, and selected data science work samples.

### Quick links

- [About](/about)
- [CV](/cv)
- [Contact](/contact)

### Latest posts

{% if site.posts.size > 0 %}
<ul>
  {% for post in site.posts limit:8 %}
    <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>
{% else %}
_No posts published yet._
{% endif %}
