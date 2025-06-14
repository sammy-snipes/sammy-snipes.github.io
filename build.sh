#!/bin/bash

for file in tex/**/*.tex; do
  reldir=$(dirname "$file" | sed 's|^tex/||')
  
  filename=$(basename -- "$file")
  name="${filename%.*}"
  
  mkdir -p "_includes/tex/$reldir"
  
  echo "Converting $file to _includes/tex/$reldir/$name.html"
    pandoc "$file" \
      -f latex \
      -t html \
      --template=assets/template/custom.html \
      --katex \
      --css=assets/css/style.css \
      --highlight-style=assets/template/cat.theme \
      -o "_includes/tex/$reldir/$name.html"
  
  dirbase=$(basename "$reldir")
  mkdir -p "_posts/$reldir"
  cat > "_posts/$reldir/$name.md" << EOF
---
layout: post
title: "$dirbase"
permalink: /$reldir/$name/
categories: latex
---
{% include tex/$reldir/$name.html %}
EOF
done

