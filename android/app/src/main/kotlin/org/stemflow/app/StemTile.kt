package org.stemflow.app

import android.content.Intent
import android.service.quicksettings.Tile
import android.service.quicksettings.TileService

/** Quick Settings tile: bass isolation on/off (helper source), like the menu's switch. */
class StemTile : TileService() {
    override fun onStartListening() = refresh()

    override fun onClick() {
        val svc = Intent(this, CaptureService::class.java)
        if (CaptureService.running) stopService(svc) else startForegroundService(svc)
        qsTile?.apply {
            state = if (CaptureService.running) Tile.STATE_INACTIVE else Tile.STATE_ACTIVE
            updateTile()
        }
    }

    private fun refresh() {
        qsTile?.apply {
            state = if (CaptureService.running) Tile.STATE_ACTIVE else Tile.STATE_INACTIVE
            label = "Bass isolation"
            subtitle = if (CaptureService.running) Control.describe() else "Off"
            updateTile()
        }
    }
}
