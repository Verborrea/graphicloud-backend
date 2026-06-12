#!/bin/bash
mkdir -p icons

ICONS=(
  "brain"
  "graph"
  "chart-bar"
  "chart-line"
  "file-text"
  "magnifying-glass"
  "books"
  "cpu"
  "arrows-clockwise"
  "chart-scatter"
)

for icon in "${ICONS[@]}"; do
  echo "Descargando $icon..."
  curl -s -o "icons/${icon}.svg" \
    "https://api.iconify.design/ph/${icon}.svg"
done

echo "✅ Listo — $(ls icons/*.svg | wc -l | tr -d ' ') íconos en ./icons/"