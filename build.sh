#!/bin/bash

DEBUG=false

if [[ "$1" == "--debug" ]]; then
  DEBUG=true
fi

# Process each .tex file
for file in tex/**/*.tex; do
  reldir=$(dirname "$file" | sed 's|^tex/||')
  
  filename=$(basename -- "$file")
  name="${filename%.*}"
  
  mkdir -p "_includes/tex/$reldir"
  
  echo "Converting $file to _includes/tex/$reldir/$name.html"
  pandoc "$file" \
    -f latex \
    -t html \
    --standalone \
    --katex \
    --highlight-style=pygments \
    -o "_includes/tex/$reldir/$name.html"
  
  # Clean up the generated HTML (unless debugging)
  if [ "$DEBUG" = false ]; then
    echo "Cleaning _includes/tex/$reldir/$name.html"
    python3 clean_html.py "_includes/tex/$reldir/$name.html" "_includes/tex/$reldir/$name.html"
  else
    echo "Skipping clean for _includes/tex/$reldir/$name.html (debug mode)"
  fi
  
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

