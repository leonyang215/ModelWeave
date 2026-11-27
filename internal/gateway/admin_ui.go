package gateway

import (
	"bytes"
	"embed"
	"net/http"
	"time"
)

// The console has no runtime CDN or Node dependency. Go builds (including
// the existing Docker gateway target) always contain matching UI assets.
//
//go:embed web/*
var adminAssets embed.FS

func adminHeaders(w http.ResponseWriter) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("Referrer-Policy", "no-referrer")
	w.Header().Set("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
}

func (s *Server) handleAdminUI(w http.ResponseWriter, r *http.Request) {
	adminHeaders(w)
	name := r.PathValue("name")
	if name == "" {
		name = "index.html"
	}
	switch name {
	case "index.html", "app.mjs", "utils.mjs", "styles.css", "favicon.svg":
	default:
		http.NotFound(w, r)
		return
	}
	b, err := adminAssets.ReadFile("web/" + name)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	if name == "app.mjs" || name == "utils.mjs" {
		w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
	}
	http.ServeContent(w, r, name, time.Time{}, bytes.NewReader(b))
}
