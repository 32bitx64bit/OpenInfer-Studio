package models

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// MetadataSchemaVersion is bumped whenever Scan persists new architecture
// fields that older library rows may lack (SWA patterns, hybrid interval, etc.).
// EnsureFresh rescans when the stored schema version differs.
const MetadataSchemaVersion = "9"

// EnsureFresh runs Scan when the persisted metadata schema is outdated or
// when on-disk GGUF files look newer/different than the library rows.
// It is safe to call on every application launch (cheap no-op when fresh).
func (l *Library) EnsureFresh(storedSchema string) (scanned bool, count int, reason string, err error) {
	if storedSchema != MetadataSchemaVersion {
		count, err = l.Scan()
		return true, count, "metadata schema upgraded", err
	}
	stale, why, err := l.staleReason()
	if err != nil {
		return false, 0, "", err
	}
	if !stale {
		return false, 0, "", nil
	}
	count, err = l.Scan()
	return true, count, why, err
}

// staleReason walks model directories without parsing GGUF payloads and compares
// discovered primaries against library rows (path set, size, mtime).
func (l *Library) staleReason() (stale bool, reason string, err error) {
	primaries, err := l.discoverPrimaries()
	if err != nil {
		return false, "", err
	}

	rows, err := l.db.Query(`SELECT primary_path, size_bytes, updated_at FROM models`)
	if err != nil {
		return false, "", err
	}
	defer rows.Close()

	type row struct {
		path string
		size int64
		upd  time.Time
	}
	byPath := map[string]row{}
	for rows.Next() {
		var path, updStr string
		var size int64
		if err := rows.Scan(&path, &size, &updStr); err != nil {
			return false, "", err
		}
		upd, perr := time.Parse(time.RFC3339Nano, updStr)
		if perr != nil {
			upd, _ = time.Parse(time.RFC3339, updStr)
		}
		byPath[filepath.Clean(path)] = row{path: path, size: size, upd: upd}
	}
	if err := rows.Err(); err != nil {
		return false, "", err
	}

	if len(primaries) != len(byPath) {
		return true, "model set changed", nil
	}
	for path, info := range primaries {
		r, ok := byPath[path]
		if !ok {
			return true, "new model file", nil
		}
		if info.size != r.size {
			return true, "model size changed", nil
		}
		// Allow a small skew so clock/FS rounding does not force rescans.
		if !r.upd.IsZero() && info.modTime.After(r.upd.Add(2*time.Second)) {
			return true, "model file newer than library", nil
		}
	}
	for path := range byPath {
		if _, ok := primaries[path]; !ok {
			return true, "library entry missing on disk", nil
		}
	}
	return false, "", nil
}

type primaryInfo struct {
	size    int64
	modTime time.Time
}

// discoverPrimaries mirrors Scan's directory walk + split grouping, but only
// records the primary shard path/size/mtime (no GGUF parse).
func (l *Library) discoverPrimaries() (map[string]primaryInfo, error) {
	dirs, err := l.Directories()
	if err != nil {
		return nil, err
	}
	type found struct {
		path string
		size int64
		mod  time.Time
	}
	var ggufs []found
	for _, d := range dirs {
		root := d["path"].(string)
		baseDepth := strings.Count(filepath.Clean(root), string(os.PathSeparator))
		_ = filepath.WalkDir(root, func(p string, e os.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			depth := strings.Count(filepath.Clean(p), string(os.PathSeparator)) - baseDepth
			if e.IsDir() {
				if depth > l.maxDepth {
					return filepath.SkipDir
				}
				if strings.HasPrefix(e.Name(), ".") && depth > 0 {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
				return nil
			}
			lower := strings.ToLower(e.Name())
			if strings.Contains(lower, "mmproj") || strings.Contains(lower, "mm-proj") {
				return nil
			}
			st, err := e.Info()
			if err != nil {
				return nil
			}
			ggufs = append(ggufs, found{p, st.Size(), st.ModTime().UTC()})
			return nil
		})
	}

	groups := map[string][]found{}
	for _, g := range ggufs {
		stem := g.path
		if splitSuffix.MatchString(stem) {
			stem = splitSuffix.ReplaceAllString(stem, "")
		} else {
			stem = strings.TrimSuffix(stem, ".gguf")
		}
		groups[stem] = append(groups[stem], g)
	}

	out := make(map[string]primaryInfo, len(groups))
	for _, files := range groups {
		sort.Slice(files, func(i, j int) bool { return files[i].path < files[j].path })
		p := files[0]
		out[filepath.Clean(p.path)] = primaryInfo{size: p.size, modTime: p.mod}
	}
	return out, nil
}
