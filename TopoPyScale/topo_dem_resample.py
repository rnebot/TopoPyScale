import fire
import os
from osgeo import gdal, osr


def downsample_geotiff_gdal(input_file, output_file=None, factor=2):
    """
    Reduce la resolución de un archivo GeoTIFF promediando píxeles adyacentes usando GDAL.

    Args:
        input_file (str): Ruta al archivo GeoTIFF de entrada.
        output_file (str, opcional): Ruta donde guardar el archivo GeoTIFF de salida.
                                    Si no se especifica, se genera automáticamente.
        factor (int, opcional): Factor de reducción. Por defecto es 2 (90m -> 180m).

    Returns:
        str: Ruta al archivo de salida generado.
    """
    # Si no se especifica un archivo de salida, generar uno
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}_180m.tif"

    # Abrir el archivo de entrada
    print(f"Leyendo {input_file}...")
    ds = gdal.Open(input_file)
    if ds is None:
        raise ValueError(f"No se pudo abrir el archivo {input_file}")

    # Obtener y mostrar información del archivo original
    width = ds.RasterXSize
    height = ds.RasterYSize
    projection = ds.GetProjectionRef()
    geotransform = ds.GetGeoTransform()

    # Analizar e imprimir resolución espacial original
    x_res = geotransform[1]
    y_res = abs(geotransform[5])  # abs porque suele ser negativo
    print(f"Información del archivo original:")
    print(f"  Dimensiones: {width}x{height} píxeles")
    print(f"  Resolución X: {x_res}")
    print(f"  Resolución Y: {y_res}")
    print(f"  Geotransform: {geotransform}")

    # Crear SpatialReference para analizar la proyección
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(projection)
    is_geographic = spatial_ref.IsGeographic()
    print(f"  Proyección geográfica (en grados): {is_geographic}")

    # Calcular las nuevas dimensiones
    new_width = width // factor
    new_height = height // factor

    # Calcular y configurar el nuevo geotransform con resolución ajustada
    new_geotransform = list(geotransform)
    new_geotransform[1] = geotransform[1] * factor  # Ajustar resolución X
    new_geotransform[5] = geotransform[5] * factor  # Ajustar resolución Y

    # Mostrar la nueva resolución esperada
    new_x_res = new_geotransform[1]
    new_y_res = abs(new_geotransform[5])
    print(f"\nNueva resolución esperada:")
    print(f"  Nueva dimensión: {new_width}x{new_height} píxeles")
    print(f"  Nueva resolución X: {new_x_res} (factor {factor}x)")
    print(f"  Nueva resolución Y: {new_y_res} (factor {factor}x)")
    print(f"  Nuevo geotransform: {new_geotransform}")

    # Configurar opciones para el promediado
    gdal_options = gdal.TranslateOptions(
        width=new_width,
        height=new_height,
        resampleAlg=gdal.GRA_Average,  # Método de promediado
        format='GTiff',
        creationOptions=['COMPRESS=LZW', 'PREDICTOR=2']  # Compresión LZW para archivos más pequeños
    )

    # Realizar la reducción de resolución
    print(f"\nReduciendo resolución por un factor de {factor}...")
    gdal.Translate(output_file, ds, options=gdal_options)

    # Verificar que el archivo resultante tiene la resolución correcta
    out_ds = gdal.Open(output_file, gdal.GA_Update)
    if out_ds:
        # Obtener y mostrar información del archivo resultante
        result_geotransform = out_ds.GetGeoTransform()
        result_projection = out_ds.GetProjectionRef()
        result_x_res = result_geotransform[1]
        result_y_res = abs(result_geotransform[5])

        print(f"\nResultado real:")
        print(f"  Resolución X resultante: {result_x_res}")
        print(f"  Resolución Y resultante: {result_y_res}")
        print(f"  Geotransform resultante: {result_geotransform}")

        # Verificar si la resolución es correcta, si no, corregirla
        if abs(result_x_res - new_x_res) > 1e-10 or abs(result_y_res - new_y_res) > 1e-10:
            print(f"  ¡Atención! La resolución resultante no coincide con la esperada. Corrigiendo...")
            out_ds.SetGeoTransform(new_geotransform)
            out_ds.FlushCache()

            # Verificar nuevamente
            corrected_geotransform = out_ds.GetGeoTransform()
            corrected_x_res = corrected_geotransform[1]
            corrected_y_res = abs(corrected_geotransform[5])
            print(f"  Resolución X corregida: {corrected_x_res}")
            print(f"  Resolución Y corregida: {corrected_y_res}")

        # Cerrar el dataset
        out_ds = None

    # Cerrar el dataset original
    ds = None

    print(f"\n¡Proceso completado! Archivo guardado en: {output_file}")

    # Mostrar información final usando gdalinfo
    print("\nInformación completa del archivo generado:")
    info = gdal.Info(output_file, format='text')
    print(info)

    return output_file


def batch_downsample_gdal(input_dir, output_dir=None, pattern="*.tif", factor=2):
    """
    Procesa todos los archivos GeoTIFF en un directorio usando GDAL.

    Args:
        input_dir (str): Directorio con los archivos GeoTIFF a procesar.
        output_dir (str, opcional): Directorio donde guardar los resultados.
                                   Si no se especifica, se usa el directorio de entrada.
        pattern (str, opcional): Patrón para filtrar archivos. Por defecto "*.tif".
        factor (int, opcional): Factor de reducción. Por defecto es 2 (90m -> 180m).

    Returns:
        list: Lista de archivos de salida generados.
    """
    import glob

    # Si no se especifica un directorio de salida, usar el de entrada
    if output_dir is None:
        output_dir = input_dir

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Obtener la lista de archivos que coinciden con el patrón
    input_files = glob.glob(os.path.join(input_dir, pattern))

    if not input_files:
        print(f"No se encontraron archivos con el patrón {pattern} en {input_dir}")
        return []

    output_files = []
    for input_file in input_files:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_180m.tif")

        try:
            result = downsample_geotiff_gdal(input_file, output_file, factor)
            output_files.append(result)
        except Exception as e:
            print(f"Error al procesar {input_file}: {e}")

    return output_files


# Implementación alternativa usando rioxarray y xarray
def downsample_geotiff_xarray(input_file, output_file=None, factor=2):
    """
    Reduce la resolución de un archivo GeoTIFF promediando píxeles adyacentes usando rioxarray.

    Args:
        input_file (str): Ruta al archivo GeoTIFF de entrada.
        output_file (str, opcional): Ruta donde guardar el archivo GeoTIFF de salida.
                                    Si no se especifica, se genera automáticamente.
        factor (int, opcional): Factor de reducción. Por defecto es 2 (90m -> 180m).

    Returns:
        str: Ruta al archivo de salida generado.
    """
    import rioxarray
    from rasterio.transform import Affine
    import numpy as np

    # Si no se especifica un archivo de salida, generar uno
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}_180m.tif"

    # Abrir y mostrar información del archivo original con GDAL
    ds = gdal.Open(input_file)
    if ds is None:
        raise ValueError(f"No se pudo abrir el archivo {input_file}")

    # Obtener y mostrar información del archivo original
    width = ds.RasterXSize
    height = ds.RasterYSize
    projection = ds.GetProjectionRef()
    geotransform = ds.GetGeoTransform()

    # Analizar e imprimir resolución espacial original
    x_res = geotransform[1]
    y_res = abs(geotransform[5])  # abs porque suele ser negativo
    print(f"Información del archivo original:")
    print(f"  Dimensiones: {width}x{height} píxeles")
    print(f"  Resolución X: {x_res}")
    print(f"  Resolución Y: {y_res}")
    print(f"  Geotransform: {geotransform}")

    # Analizar la proyección
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(projection)
    is_geographic = spatial_ref.IsGeographic()
    print(f"  Proyección geográfica (en grados): {is_geographic}")

    # Calcular nueva resolución esperada
    new_x_res = x_res * factor
    new_y_res = y_res * factor
    new_width = width // factor
    new_height = height // factor

    # Crear nuevo geotransform con resolución ajustada
    new_geotransform = list(geotransform)
    new_geotransform[1] = geotransform[1] * factor  # Ajustar resolución X
    new_geotransform[5] = geotransform[5] * factor  # Ajustar resolución Y

    # Mostrar la nueva resolución esperada
    print(f"\nNueva resolución esperada:")
    print(f"  Nueva dimensión: {new_width}x{new_height} píxeles")
    print(f"  Nueva resolución X: {new_x_res} (factor {factor}x)")
    print(f"  Nueva resolución Y: {new_y_res} (factor {factor}x)")
    print(f"  Nuevo geotransform: {new_geotransform}")

    # Cerrar dataset GDAL
    ds = None

    # Leer el archivo GeoTIFF usando rioxarray
    print(f"\nLeyendo {input_file} con rioxarray...")
    raster = rioxarray.open_rasterio(input_file)

    # Reducir la resolución promediando píxeles adyacentes
    print(f"Reduciendo resolución por un factor de {factor}...")
    # Redimensionar usando xarray coarsen con la función mean
    downsampled = raster.coarsen(x=factor, y=factor, boundary='trim').mean()

    # Actualizar los metadatos de transformación para reflejar la nueva resolución
    old_transform = raster.rio.transform()
    new_transform = Affine(
        old_transform[0] * factor,  # a: ancho del pixel (resolución X)
        old_transform[1],  # b: rotación de fila (normalmente 0)
        old_transform[2],  # c: x_origen
        old_transform[3],  # d: rotación de columna (normalmente 0)
        old_transform[4] * factor,  # e: alto del pixel (resolución Y, normalmente negativo)
        old_transform[5]  # f: y_origen
    )

    # Mostrar transformación
    print(f"Transformación original rioxarray: {old_transform}")
    print(f"Nueva transformación rioxarray: {new_transform}")

    # Guardar el resultado como un nuevo archivo GeoTIFF
    print(f"Guardando resultado en {output_file}...")
    downsampled.rio.write_crs(raster.rio.crs, inplace=True)
    downsampled.rio.write_transform(new_transform, inplace=True)
    downsampled.rio.to_raster(output_file)

    # Verificar que el archivo resultante tiene la resolución correcta usando GDAL
    out_ds = gdal.Open(output_file, gdal.GA_Update)
    if out_ds:
        # Obtener y mostrar información del archivo resultante
        result_geotransform = out_ds.GetGeoTransform()
        result_projection = out_ds.GetProjectionRef()
        result_x_res = result_geotransform[1]
        result_y_res = abs(result_geotransform[5])

        print(f"\nResultado real:")
        print(f"  Resolución X resultante: {result_x_res}")
        print(f"  Resolución Y resultante: {result_y_res}")
        print(f"  Geotransform resultante: {result_geotransform}")

        # Verificar si la resolución es correcta, si no, corregirla
        if abs(result_x_res - new_x_res) > 1e-10 or abs(result_y_res - new_y_res) > 1e-10:
            print(f"  ¡Atención! La resolución resultante no coincide con la esperada. Corrigiendo...")
            # Aplicar el geotransform correcto
            out_ds.SetGeoTransform(new_geotransform)
            # Asegurar que la proyección es correcta
            out_ds.SetProjection(projection)  # Usar la proyección original
            out_ds.FlushCache()

            # Verificar nuevamente
            corrected_geotransform = out_ds.GetGeoTransform()
            corrected_x_res = corrected_geotransform[1]
            corrected_y_res = abs(corrected_geotransform[5])
            print(f"  Resolución X corregida: {corrected_x_res}")
            print(f"  Resolución Y corregida: {corrected_y_res}")

        # Cerrar el dataset
        out_ds = None

    print(f"\n¡Proceso completado! Archivo guardado en: {output_file}")

    # Mostrar información final usando gdalinfo
    print("\nInformación completa del archivo generado:")
    info = gdal.Info(output_file, format='text')
    print(info)

    return output_file


def batch_downsample_xarray(input_dir, output_dir=None, pattern="*.tif", factor=2):
    """
    Procesa todos los archivos GeoTIFF en un directorio usando rioxarray.

    Args:
        input_dir (str): Directorio con los archivos GeoTIFF a procesar.
        output_dir (str, opcional): Directorio donde guardar los resultados.
                                   Si no se especifica, se usa el directorio de entrada.
        pattern (str, opcional): Patrón para filtrar archivos. Por defecto "*.tif".
        factor (int, opcional): Factor de reducción. Por defecto es 2 (90m -> 180m).

    Returns:
        list: Lista de archivos de salida generados.
    """
    import glob

    # Si no se especifica un directorio de salida, usar el de entrada
    if output_dir is None:
        output_dir = input_dir

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Obtener la lista de archivos que coinciden con el patrón
    input_files = glob.glob(os.path.join(input_dir, pattern))

    if not input_files:
        print(f"No se encontraron archivos con el patrón {pattern} en {input_dir}")
        return []

    output_files = []
    for input_file in input_files:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_180m.tif")

        try:
            result = downsample_geotiff_xarray(input_file, output_file, factor)
            output_files.append(result)
        except Exception as e:
            print(f"Error al procesar {input_file}: {e}")

    return output_files


# Permitir ejecutar el script desde la línea de comandos
if __name__ == "__main__":
    fire.Fire({
        "downsample_gdal": downsample_geotiff_gdal,
        "batch_gdal": batch_downsample_gdal,
        "downsample_xarray": downsample_geotiff_xarray,
        "batch_xarray": batch_downsample_xarray
    })