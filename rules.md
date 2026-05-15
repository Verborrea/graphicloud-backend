# Orden:

1. Subir archivos iniciales.
   1. Si solo hay un archivo => 1 nube global en el centro 0/0
   2. Si hay dos archivos => 2 locales + 1 global, pero en lugar de normalizar, que la distancia entre nubes sea el coseno... o está bien normalizadas?
   3. Si hay n arhchivos => n locales + 1 global
2. Al cambiar los documentos, volver a generar las glcouds
3. Al cambiar de modo, alternar entre las gclouds mostradas, no llamar de nuevo al algoritmo.
4. Al cambiar las capas, solo alteraciones visuales
5. Al cambiar las preferencias, volver a generar las glclouds, pero manteniendo íconos...
6. Al cambiar la configuración, volver a generar las glcouds

Entonces, los docs y la configuración resetean todo. De donde sacan todo? hay que guardar el resultado de la api en un state.

Las preferencias vuelven a llamar al algoritmo pero ahora usa íconos. Entonces el algoritmo debe de aceptar cualquier forma, no el texto, sino la forma...

1 => Subir archivos
2 => Guardar Resultados crudos?

Un nodo debería guardar:

```
{
   texts: string[]
   x: number
   y: number
   w: number
   h: number
   icon?: string
   fontSize?: number
}
```

y yap creo yo... el fontsize creo que sí es necesario para dibujarlo en el componente que dibuja las nubes. Texts es un array porq si es un ícono y hay que generar más íconos a partir de él, habría que guardar los conceptos que tiene no? El fontSize igual representa al score así que no sé si guardar los dos.

La nube guarda su id, offsets, color y su lista de nodos.
