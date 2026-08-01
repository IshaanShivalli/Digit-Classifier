import pygame
import torch
import torch.nn as nn
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 900, 620
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Digit Classifier")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
BLUE = (52, 152, 219)
GREEN = (46, 204, 113)
RED = (231, 76, 60)

# Grid settings
GRID_SIZE = 28
CELL_SIZE = 18
CANVAS_SIZE = GRID_SIZE * CELL_SIZE
CANVAS_X = 40
CANVAS_Y = 100
CANVAS_RECT = pygame.Rect(CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE)

# Define the model to match the training script
model = nn.Sequential(
    nn.Linear(28 * 28, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Load the model
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

try:
    state_dict = torch.load('digit_model.pth', map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Could not load digit_model.pth: {e}")
    print("Using untrained model.")

# Canvas for drawing as a 28x28 grid matching MNIST input
canvas = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
canvas_surface = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))

# Drawing variables
drawing = False
last_cell = None
prediction = None
confidence = None

# Font
font_large = pygame.font.Font(None, 48)
font_medium = pygame.font.Font(None, 32)
font_small = pygame.font.Font(None, 24)


def preprocess_image(image_array):
    """Preprocess the drawn image for the model."""
    image = image_array.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image).reshape(1, -1).to(device)
    return image_tensor


def classify_digit():
    """Classify the drawn digit."""
    global prediction, confidence

    image_tensor = preprocess_image(canvas)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        pred_digit = torch.argmax(probabilities, dim=1).item()
        conf = probabilities[0, pred_digit].item() * 100

    prediction = pred_digit
    confidence = conf


def draw_line(start_pos, end_pos):
    """Draw a continuous line between two grid cells."""
    x0, y0 = start_pos
    x1, y1 = end_pos
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        canvas[y0, x0] = 255
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def draw_canvas():
    """Render the grid-based drawing canvas."""
    canvas_surface.fill(BLACK)

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if canvas[y, x] > 0:
                pygame.draw.rect(
                    canvas_surface,
                    WHITE,
                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                )

    for line in range(GRID_SIZE + 1):
        pygame.draw.line(canvas_surface, GRAY, (line * CELL_SIZE, 0), (line * CELL_SIZE, CANVAS_SIZE), 1)
        pygame.draw.line(canvas_surface, GRAY, (0, line * CELL_SIZE), (CANVAS_SIZE, line * CELL_SIZE), 1)

    screen.blit(canvas_surface, (CANVAS_X, CANVAS_Y))
    pygame.draw.rect(screen, BLACK, CANVAS_RECT, 2)


def draw_ui():
    """Draw the interface."""
    title = font_large.render("Digit Classifier", True, BLACK)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))

    instruction = font_small.render("Draw one digit on the grid below.", True, GRAY)
    screen.blit(instruction, (CANVAS_X, 66))

    clear_btn = pygame.Rect(610, 120, 230, 54)
    classify_btn = pygame.Rect(610, 190, 230, 54)

    pygame.draw.rect(screen, LIGHT_GRAY, clear_btn)
    pygame.draw.rect(screen, GRAY, clear_btn, 2)
    clear_text = font_small.render("Clear Canvas", True, BLACK)
    screen.blit(clear_text, (clear_btn.centerx - clear_text.get_width() // 2, clear_btn.centery - clear_text.get_height() // 2))

    pygame.draw.rect(screen, BLUE, classify_btn)
    classify_text = font_small.render("Classify Digit", True, WHITE)
    screen.blit(classify_text, (classify_btn.centerx - classify_text.get_width() // 2, classify_btn.centery - classify_text.get_height() // 2))

    if prediction is not None:
        result_box = pygame.Rect(610, 280, 230, 190)
        pygame.draw.rect(screen, LIGHT_GRAY, result_box)
        pygame.draw.rect(screen, BLACK, result_box, 2)

        pred_label = font_small.render("Prediction:", True, BLACK)
        screen.blit(pred_label, (630, 305))

        pred_text = font_large.render(str(prediction), True, BLUE)
        screen.blit(pred_text, (result_box.centerx - pred_text.get_width() // 2, 340))

        conf_label = font_small.render("Confidence:", True, BLACK)
        screen.blit(conf_label, (630, 410))

        conf_text = font_small.render(f"{confidence:.1f}%", True, GREEN)
        screen.blit(conf_text, (result_box.centerx - conf_text.get_width() // 2, 440))

    return clear_btn, classify_btn


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            clear_btn, classify_btn = draw_ui()

            if CANVAS_RECT.collidepoint(mouse_x, mouse_y):
                drawing = True
                last_cell = None
                cell_x = (mouse_x - CANVAS_X) // CELL_SIZE
                cell_y = (mouse_y - CANVAS_Y) // CELL_SIZE
                if 0 <= cell_x < GRID_SIZE and 0 <= cell_y < GRID_SIZE:
                    canvas[cell_y, cell_x] = 255
                    last_cell = (cell_x, cell_y)

            if clear_btn.collidepoint(event.pos):
                canvas[:, :] = 0
                prediction = None
                confidence = None

            if classify_btn.collidepoint(event.pos):
                classify_digit()

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_cell = None

        elif event.type == pygame.MOUSEMOTION and drawing:
            mouse_x, mouse_y = event.pos
            cell_x = (mouse_x - CANVAS_X) // CELL_SIZE
            cell_y = (mouse_y - CANVAS_Y) // CELL_SIZE

            if 0 <= cell_x < GRID_SIZE and 0 <= cell_y < GRID_SIZE:
                if last_cell is None:
                    canvas[cell_y, cell_x] = 255
                    last_cell = (cell_x, cell_y)
                else:
                    draw_line(last_cell, (cell_x, cell_y))
                    last_cell = (cell_x, cell_y)

    screen.fill(WHITE)
    clear_btn, classify_btn = draw_ui()
    draw_canvas()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()