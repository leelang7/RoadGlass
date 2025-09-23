
package com.ultralytics.ultralytics_yolo_example

import android.content.ContentValues
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.OutputStream
import android.graphics.Bitmap
import android.view.PixelCopy
import android.os.Handler
import android.os.Looper
import java.io.ByteArrayOutputStream
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import android.view.Surface

class MainActivity : FlutterActivity() {
    private val CHANNEL = "app.gallery_saver"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "saveImage" -> {
                    val pathArg = call.argument<String>("path")
                    if (pathArg.isNullOrBlank()) {
                        result.error("ARG_ERROR", "Missing 'path' argument", null)
                        return@setMethodCallHandler
                    }
                    try {
                        val savedUri = saveImageToGallery(pathArg)
                        if (savedUri != null) {
                            result.success(mapOf("isSuccess" to true, "uri" to savedUri.toString(), "path" to pathArg))
                        } else {
                            result.error("SAVE_FAILED", "Failed to save image", null)
                        }
                    } catch (e: Exception) {
                        result.error("EXCEPTION", e.message, null)
                    }
                }
                else -> result.notImplemented()
            }
        }
        // Native camera capture channel: capture directly from the camera TextureView surface (no Flutter overlays)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "app.camera_capture").setMethodCallHandler { call, result ->
            if (call.method == "capture") {
                val textureView = findFirstTextureView(window.decorView)
                if (textureView == null || !textureView.isAvailable) {
                    result.error("NO_TEXTURE", "TextureView not found or not available", null)
                    return@setMethodCallHandler
                }

                val w = textureView.width.coerceAtLeast(1)
                val h = textureView.height.coerceAtLeast(1)
                val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

                // Use the TextureView's Surface for PixelCopy (captures native camera frame only)
                val surface = Surface(textureView.surfaceTexture)
                try {
                    PixelCopy.request(surface, bmp, { copyResult ->
                        try {
                            if (copyResult == PixelCopy.SUCCESS) {
                                val stream = ByteArrayOutputStream()
                                // JPEG is smaller and widely supported
                                bmp.compress(Bitmap.CompressFormat.JPEG, 92, stream)
                                result.success(stream.toByteArray())
                            } else {
                                result.error("COPY_FAIL", "PixelCopy failed: $copyResult", null)
                            }
                        } finally {
                            bmp.recycle()
                            surface.release()
                        }
                    }, Handler(Looper.getMainLooper()))
                } catch (e: Throwable) {
                    bmp.recycle()
                    surface.release()
                    result.error("EX", e.message, null)
                }
            } else {
                result.notImplemented()
            }
        }
    }

    private fun saveImageToGallery(sourcePath: String): Uri? {
        val srcFile = File(sourcePath)
        if (!srcFile.exists()) return null

        val fileName = srcFile.name.ifBlank { "capture_${System.currentTimeMillis()}.png" }
        val mimeType = when {
            fileName.endsWith(".png", true) -> "image/png"
            fileName.endsWith(".jpg", true) || fileName.endsWith(".jpeg", true) -> "image/jpeg"
            else -> "image/png"
        }

        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/RoadGlass")
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val resolver = applicationContext.contentResolver
            val collection = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
            val itemUri = resolver.insert(collection, values) ?: return null
            resolver.openOutputStream(itemUri)?.use { out ->
                FileInputStream(srcFile).use { input -> input.copyTo(out) }
            }
            values.clear()
            values.put(MediaStore.MediaColumns.IS_PENDING, 0)
            resolver.update(itemUri, values, null, null)
            itemUri
        } else {
            // Android 9 (Pie) 이하: 퍼블릭 Pictures/RoadGlass 경로에 복사 후 미디어 스캔
            val picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
            val targetDir = File(picturesDir, "RoadGlass")
            if (!targetDir.exists()) targetDir.mkdirs()
            val targetFile = File(targetDir, fileName)
            FileInputStream(srcFile).use { input ->
                FileOutputStream(targetFile).use { output: OutputStream ->
                    input.copyTo(output)
                }
            }
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DATA, targetFile.absolutePath)
                put(MediaStore.Images.Media.MIME_TYPE, mimeType)
            }
            applicationContext.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        }
    }
    private fun findFirstTextureView(root: View): TextureView? {
        if (root is TextureView) return root
        if (root is ViewGroup) {
            for (i in 0 until root.childCount) {
                val found = findFirstTextureView(root.getChildAt(i))
                if (found != null) return found
            }
        }
        return null
    }
}
