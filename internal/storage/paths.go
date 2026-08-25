// Package storage validates filesystem paths used by the backend: every path
// arriving over the API must resolve inside an allowed root.
package storage

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ValidateInside resolves p (which may be relative to root) and requires the
// result to stay inside root after symlink resolution of the root. It rejects
// traversal such as "../../etc/passwd".
//
// The destination often does not exist yet (quantize output, imports). Root is
// EvalSymlink'd (macOS /var → /private/var, Windows 8.3 names); p must be
// resolved the same way or a nested path looks like it escaped.
func ValidateInside(root, p string) (string, error) {
	if p == "" {
		return "", fmt.Errorf("empty path")
	}
	absRoot, err := absResolved(root)
	if err != nil {
		return "", err
	}
	abs := p
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(absRoot, p)
	}
	clean, err := filepath.Abs(filepath.Clean(abs))
	if err != nil {
		return "", err
	}
	resolved, err := resolveExisting(clean)
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(canonical(absRoot), canonical(resolved))
	if err != nil || !relInside(rel) {
		return "", fmt.Errorf("path %q escapes managed root %q", p, root)
	}
	return clean, nil
}

func relInside(rel string) bool {
	if rel == "." {
		return true
	}
	rel = filepath.ToSlash(rel)
	return rel != ".." && !strings.HasPrefix(rel, "../")
}

// canonical strips Windows \\?\ prefixes so Rel can compare EvalSymlinks output
// with Abs paths.
func canonical(p string) string {
	p = filepath.Clean(p)
	switch {
	case strings.HasPrefix(p, `\\?\UNC\`):
		return `\\` + p[len(`\\?\UNC\`):]
	case strings.HasPrefix(p, `\\?\`):
		return p[len(`\\?\`):]
	}
	return p
}

func absResolved(p string) (string, error) {
	abs, err := filepath.Abs(p)
	if err != nil {
		return "", err
	}
	if resolved, err := filepath.EvalSymlinks(abs); err == nil {
		return resolved, nil
	}
	return abs, nil
}

// resolveExisting EvalSymlinks the longest existing prefix of p and reattaches
// missing trailing components (the output file may not exist yet).
func resolveExisting(p string) (string, error) {
	p = filepath.Clean(p)
	cur := p
	var missing []string
	for {
		if resolved, err := filepath.EvalSymlinks(cur); err == nil {
			for i := len(missing) - 1; i >= 0; i-- {
				resolved = filepath.Join(resolved, missing[i])
			}
			return resolved, nil
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			return p, nil
		}
		missing = append(missing, filepath.Base(cur))
		cur = parent
	}
}

// SafeJoin joins elements to root and validates containment in one call.
func SafeJoin(root string, elems ...string) (string, error) {
	return ValidateInside(root, filepath.Join(elems...))
}

// AtomicWrite writes data to a temp file in the same directory and renames it
// over dst, so readers never observe a partially written file.
func AtomicWrite(dst string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(dst)
	tmp, err := os.CreateTemp(dir, ".atomic-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Chmod(perm); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	return os.Rename(tmpName, dst)
}
