package quantize

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/openinfer/openinfer-studio/internal/gguf"
	"github.com/openinfer/openinfer-studio/internal/storage"
)

// IMatrix is a reusable importance-matrix file tied to a source model.
type IMatrix struct {
	ID            string `json:"id"`
	SourceModelID string `json:"source_model_id"`
	Path          string `json:"path"`
	Format        string `json:"format"`
	DatasetLabel  string `json:"dataset_label"`
	NChunks       int    `json:"n_chunks"`
	Origin        string `json:"origin"`
	CreatedAt     string `json:"created_at"`
}

func imatrixFormat(path string) string {
	if strings.EqualFold(filepath.Ext(path), ".dat") {
		return "dat"
	}
	return "gguf"
}

func (m *Manager) ListIMatrices(modelID string) ([]IMatrix, error) {
	q := `SELECT id,source_model_id,path,format,dataset_label,n_chunks,origin,created_at FROM imatrices`
	args := []any{}
	if modelID != "" {
		q += ` WHERE source_model_id = ?`
		args = append(args, modelID)
	}
	q += ` ORDER BY created_at DESC`
	rows, err := m.db.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []IMatrix
	for rows.Next() {
		var im IMatrix
		if err := rows.Scan(&im.ID, &im.SourceModelID, &im.Path, &im.Format, &im.DatasetLabel, &im.NChunks, &im.Origin, &im.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, im)
	}
	if out == nil {
		out = []IMatrix{}
	}
	return out, nil
}

func (m *Manager) getIMatrix(id string) (*IMatrix, error) {
	var im IMatrix
	err := m.db.QueryRow(`SELECT id,source_model_id,path,format,dataset_label,n_chunks,origin,created_at FROM imatrices WHERE id = ?`, id).
		Scan(&im.ID, &im.SourceModelID, &im.Path, &im.Format, &im.DatasetLabel, &im.NChunks, &im.Origin, &im.CreatedAt)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("imatrix %s not found", id)
	}
	if err != nil {
		return nil, err
	}
	return &im, nil
}

func (m *Manager) insertIMatrix(im IMatrix) error {
	_, err := m.db.Exec(`INSERT INTO imatrices(id,source_model_id,path,format,dataset_label,n_chunks,origin,created_at)
		VALUES (?,?,?,?,?,?,?,?)`,
		im.ID, im.SourceModelID, im.Path, im.Format, im.DatasetLabel, im.NChunks, im.Origin, im.CreatedAt)
	return err
}

// ImportIMatrix copies (or registers) an existing imatrix file.
func (m *Manager) ImportIMatrix(modelID, srcPath, label string) (*IMatrix, error) {
	abs, err := filepath.Abs(srcPath)
	if err != nil {
		return nil, err
	}
	if st, err := os.Stat(abs); err != nil || st.IsDir() {
		return nil, fmt.Errorf("invalid imatrix path")
	}
	id := uuid.NewString()
	format := imatrixFormat(abs)
	destName := id + filepath.Ext(abs)
	if format == "gguf" && filepath.Ext(abs) == "" {
		destName = id + ".gguf"
	}
	dest, err := storage.SafeJoin(m.layout.QuantIMatrices, destName)
	if err != nil {
		return nil, err
	}
	if err := copyFile(abs, dest); err != nil {
		return nil, err
	}
	if label == "" {
		label = filepath.Base(abs)
	}
	im := IMatrix{
		ID: id, SourceModelID: modelID, Path: dest, Format: format,
		DatasetLabel: label, Origin: "imported", CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
	}
	if err := m.insertIMatrix(im); err != nil {
		return nil, err
	}
	return &im, nil
}

// DeleteIMatrix removes the DB row and, when the file lives under the managed
// imatrices directory, the file itself.
func (m *Manager) DeleteIMatrix(id string, deleteFile bool) error {
	im, err := m.getIMatrix(id)
	if err != nil {
		return err
	}
	if deleteFile {
		clean := filepath.Clean(im.Path)
		root := filepath.Clean(m.layout.QuantIMatrices)
		if clean == root || strings.HasPrefix(clean, root+string(os.PathSeparator)) {
			_ = os.Remove(clean)
		}
	}
	_, err = m.db.Exec(`DELETE FROM imatrices WHERE id = ?`, id)
	return err
}

func (m *Manager) recordGeneratedIMatrix(modelID, path, label string, chunks int, origin string) (*IMatrix, error) {
	im := IMatrix{
		ID: uuid.NewString(), SourceModelID: modelID, Path: path,
		Format: imatrixFormat(path), DatasetLabel: label, NChunks: chunks,
		Origin: origin, CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
	}
	return &im, m.insertIMatrix(im)
}

func imatrixChunkCount(path string) int {
	if fileSize(path) <= 0 {
		return 0
	}
	md, err := gguf.ParseFile(path)
	if err != nil || md == nil || md.Raw == nil {
		return 0
	}
	return kvAsInt(md.Raw["imatrix.chunk_count"])
}

func kvAsInt(v any) int {
	switch n := v.(type) {
	case uint8:
		return int(n)
	case uint16:
		return int(n)
	case uint32:
		return int(n)
	case uint64:
		if n > uint64(^uint(0)>>1) {
			return 0
		}
		return int(n)
	case int8:
		return int(n)
	case int16:
		return int(n)
	case int32:
		return int(n)
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}
