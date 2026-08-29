package orchestrate

import "path/filepath"

// unixAbs maps a POSIX absolute path onto an OS-absolute path so
// Invocation.Validate (filepath.IsAbs) accepts fixtures on Windows.
func unixAbs(p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	return "C:" + filepath.FromSlash(p)
}
