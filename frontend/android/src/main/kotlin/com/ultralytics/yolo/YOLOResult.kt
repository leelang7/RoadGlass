// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF

data class YOLOResult(
    val origShape: Size,
    val boxes: List<Box> = emptyList(),
    val masks: Masks? = null,
    val probs: Probs? = null,
    val keypointsList: List<Keypoints> = emptyList(),
    val obb: List<OBBResult> = emptyList(),
    val annotatedImage: Bitmap? = null,
    val speed: Double,
    val fps: Double? = null,
    val originalImage: Bitmap? = null,
    val names: List<String>
)

data class Box(
    var index: Int,
    var cls: String,
    var conf: Float,
    var xywh: RectF,    // Real image coordinates
    var xywhn: RectF    // Normalized coordinates (0~1)
)

data class Masks(
    val masks: List<List<List<Float>>>, // Individual probability maps (matrix list)
    val combinedMask: Bitmap?           // Combined mask image
)

data class Probs(
    var top1: String,
    var top5: List<String>,
    var top1Conf: Float,
    var top5Confs: List<Float>,
    var top1Index: Int
)

data class Keypoints(
    val xyn: List<Pair<Float, Float>>,
    val xy: List<Pair<Float, Float>>,
    val conf: List<Float>
)

data class OBBResult(
    val box: OBB,
    val confidence: Float,
    val cls: String,
    val index: Int
)

data class Size(val width: Int, val height: Int)


fun YOLOResult.toDetectionsMap(): List<Map<String, Any>> {
    val detections = mutableListOf<Map<String, Any>>()
    this.boxes.forEach { box ->
        val className = box.cls
        val confidence = box.conf
        val bbox = mapOf(
            "x" to box.xywh.left,
            "y" to box.xywh.top,
            "width" to box.xywh.width(),
            "height" to box.xywh.height()
        )
        detections.add(
            mapOf(
                "className" to className,
                "confidence" to confidence,
                "bbox" to bbox
            )
        )
    }
    return detections
}
