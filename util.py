def get_center_of_mass(points):
        center_pheromone_x = 0.0
        center_pheromone_y = 0.0
        center_pheromone_z = 0.0
        center_pheromone_f = 0.0
        total_pheromone = 0.0
        for point in points:
            pheromone = point.get_pheromone()
            center_pheromone_x += point.get_x() * pheromone
            center_pheromone_y += point.get_y() * pheromone
            center_pheromone_z += point.get_z() * pheromone
            center_pheromone_f += point.get_f() * pheromone
            total_pheromone += pheromone
        if total_pheromone == 0:
            return None
        return [
            center_pheromone_x / total_pheromone,
            center_pheromone_y / total_pheromone,
            center_pheromone_z / total_pheromone,
            center_pheromone_f / total_pheromone,
        ]